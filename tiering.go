package main

import (
	"strings"
)

// ResolveTieringAndEffort handles dynamic tiering and effort mapping:
// 1. Prioritizes ultra-fast gemini-3.8-flash-tiered with prompt caching for claude-* models.
// 2. Maps *-tiered models based on output_config.effort or auxiliary requests.
// 3. Configures thinking budget:
//   - low -> 2048 (gemini-3.8-flash-low)
//   - medium -> 8192 (gemini-3.8-flash-medium, safe production default)
//   - high -> 16384 (gemini-3.8-flash-high)
//   - xhigh / max / ultracode -> -1 (gemini-3.8-flash-tiered with Native Google ModeAuto)
//   - auxiliary -> 2048 (gemini-3.8-flash-low, eliminates 20s stalls on metadata)
func ResolveTieringAndEffort(claudeReqJSON map[string]interface{}, reqClass string) (string, int) {
	model, _ := claudeReqJSON["model"].(string)

	// 1. Explicit native models preservation
	if model == "claude-sonnet-4-6" || model == "claude-opus-4-6-thinking" || model == "gpt-oss-120b-medium" {
		// Native 3P models supported by CloudCode - keep if explicitly requested
	} else if strings.HasPrefix(model, "claude-opus") || strings.HasPrefix(model, "claude-3-opus") ||
		strings.HasPrefix(model, "claude-fable") || strings.HasPrefix(model, "claude-mythos") {
		// Canonical mapping matching Python relay-proxy: Opus maps to gemini-3.8-flash-tiered
		model = "gemini-3.8-flash-tiered"
	} else if strings.HasPrefix(model, "claude-sonnet") || strings.HasPrefix(model, "claude-3-5-sonnet") || strings.HasPrefix(model, "claude-3-7-sonnet") {
		model = "gemini-3.8-flash-medium"
	} else if strings.HasPrefix(model, "claude-haiku") || strings.HasPrefix(model, "claude-3-5-haiku") {
		model = "gemini-2.5-flash"
	}

	effort := ""
	if outCfg, ok := claudeReqJSON["output_config"].(map[string]interface{}); ok {
		if eff, ok := outCfg["effort"].(string); ok {
			effort = strings.ToLower(eff)
		}
	}

	// 2. Dynamic sub-tier and budget mapping for Gemini 3.8 / tiered models
	budget := 8192 // default medium

	if strings.HasPrefix(model, "gemini-3.8-flash") || model == "gemini-3.8-flash-tiered" {
		// Opus/Fable/Mythos were already mapped to the top "tiered"
		// (ModeAuto) class above specifically because they're the
		// flagship tier — but Sonnet maps to "-medium" and enters this
		// exact same switch too, so without remembering which one this
		// was, the no-effort default case below silently downgraded every
		// Opus-class request with no explicit output_config.effort
		// (anything not setting CLAUDE_CODE_EFFORT_LEVEL) from "tiered"
		// down to plain "-medium" with a capped 8192 budget.
		wasTieredClass := model == "gemini-3.8-flash-tiered"
		base := "gemini-3.8-flash"
		if reqClass == "auxiliary" {
			model = base + "-low"
			budget = 2048
		} else {
			switch effort {
			case "low":
				if wasTieredClass {
					// Opus/Fable/Mythos never degrade all the way to the
					// bottom tier even on an explicit "low" — same floor
					// the rest of this class gets below.
					model = base + "-medium"
					budget = 8192
				} else {
					model = base + "-low"
					budget = 2048
				}
			case "medium", "high", "xhigh", "max", "ultracode":
				if wasTieredClass {
					// Asking for MORE effort on an Opus-class model used to
					// map "high" to a capped 16384-token budget — strictly
					// LESS thinking than the unlimited ModeAuto tier that
					// same request got with NO effort specified at all, so
					// explicitly requesting more reasoning quality made it
					// worse. Every non-"low" level funnels into the same
					// full tiered/ModeAuto treatment for this class; Google's
					// own adaptive thinking budget already scales with the
					// request, which an artificial fixed cap can't do.
					model = "gemini-3.8-flash-tiered"
					budget = -1
				} else if effort == "medium" {
					model = base + "-medium"
					budget = 8192
				} else if effort == "high" {
					model = base + "-high"
					budget = 16384
				} else {
					model = "gemini-3.8-flash-tiered"
					budget = -1 // Native Google ModeAuto
				}
			default:
				if wasTieredClass {
					model = "gemini-3.8-flash-tiered"
					budget = -1
				} else {
					// Default when no explicit effort is passed: balanced medium
					model = base + "-medium"
					budget = 8192
				}
			}
		}
	} else if strings.HasSuffix(model, "-tiered") {
		base := strings.TrimSuffix(model, "-tiered")
		if reqClass == "auxiliary" {
			model = base + "-low"
			budget = 2048
		} else {
			switch effort {
			case "low":
				model = base + "-low"
				budget = 2048
			case "medium":
				model = base + "-medium"
				budget = 8192
			case "high":
				model = base + "-high"
				budget = 16384
			case "xhigh", "max", "ultracode":
				model = base + "-tiered"
				budget = -1
			default:
				model = base + "-medium"
				budget = 8192
			}
		}
	} else {
		// Non-tiered models thinking budget
		if reqClass == "auxiliary" {
			budget = 2048
		} else {
			switch effort {
			case "low":
				budget = 2048
			case "medium":
				budget = 8192
			case "high":
				budget = 16384
			case "xhigh", "max", "ultracode":
				budget = -1
			default:
				if strings.HasSuffix(model, "-high") {
					budget = 16384
				} else if strings.HasSuffix(model, "-low") {
					budget = 2048
				} else if strings.HasSuffix(model, "-medium") {
					budget = 8192
				} else if model == "gemini-2.5-flash" {
					// Haiku's whole point is a fast, cheap response — none
					// of the suffix checks above match this model name, so
					// it fell through to the outer "budget := 8192" default
					// set at the top of this function, forcing a medium
					// thinking pass (and its latency/cost) onto every
					// Haiku request that doesn't explicitly ask for one.
					budget = 0
				}
			}
		}
	}

	// Standard Anthropic "thinking" object — Claude Code itself never sends
	// this (it uses the proprietary output_config.effort above), but any
	// generic Anthropic SDK client or non-Claude-Code caller would. Only
	// applies when effort didn't already decide the budget, so Claude
	// Code's own requests are completely unaffected.
	if effort == "" {
		if th, ok := claudeReqJSON["thinking"].(map[string]interface{}); ok {
			switch th["type"] {
			case "disabled":
				budget = 0
			case "enabled":
				if bt, ok := th["budget_tokens"].(float64); ok && bt > 0 {
					budget = int(bt)
				}
			case "adaptive":
				// Not a type this codebase has independently confirmed in
				// Anthropic's own public docs, but harmless either way: if
				// no real client ever sends it, this case just never
				// triggers. ModeAuto is the closest Gemini-side equivalent
				// to "let the model decide its own thinking depth".
				budget = -1
			}
		}
	}

	claudeReqJSON["model"] = model
	return model, budget
}
