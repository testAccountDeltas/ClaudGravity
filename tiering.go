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
func ResolveTieringAndEffort(claudeReqJSON map[string]interface{}, reqClass string, agentKey string) (string, int) {
	model, _ := claudeReqJSON["model"].(string)

	// An audit caught that this function only ever checked reqClass ==
	// "auxiliary" for the cheap/fast tier — reqClass == "subagent" (the
	// generic Task-tool fallback) was never special-cased at all, and a
	// lightweight Explore subagent (ParseAgentInfo's agentKey ==
	// "sub:Explore") inherited the SAME heavy ModeAuto/unlimited-thinking
	// budget as the main Opus-class thread, adding 30-60s of thinking
	// latency to what should be a quick, cheap search. Explore (and any
	// other investigation-only subagent with no edit/write tools, by
	// convention named the same way) gets the auxiliary-style cheap tier
	// treatment below instead.
	// An audit caught this only ever matched "sub:Explore" literally —
	// other read-only, investigation-only subagent types (doc lookup,
	// trivial config edits, a read-only transcript scan) got the SAME
	// heavy ModeAuto/unlimited-thinking treatment as a full Opus-class
	// coding subagent, for the same reason Explore originally needed
	// this fix. Scoped to the specific known-lightweight types rather
	// than "any subagent", since most subagent types (general-purpose,
	// code-reviewer, etc.) genuinely do need full reasoning capacity.
	isLightweightSubagent := false
	switch strings.ToLower(agentKey) {
	case "sub:explore", "sub:claude-code-guide", "sub:statusline-setup", "sub:fewer-permission-prompts":
		isLightweightSubagent = true
	}

	// 1. Explicit native models preservation
	if model == "claude-sonnet-4-6" || model == "claude-opus-4-6-thinking" || model == "gpt-oss-120b-medium" {
		// Native 3P models supported by CloudCode - keep if explicitly requested
	} else if strings.HasPrefix(model, "claude-opus") || strings.HasPrefix(model, "claude-3-opus") ||
		strings.HasPrefix(model, "claude-fable") || strings.HasPrefix(model, "claude-mythos") ||
		model == "opus" || model == "fable" || model == "mythos" {
		// Canonical mapping matching Python relay-proxy: Opus maps to gemini-3.8-flash-tiered
		// An audit caught that Claude Code's Agent/Task tool schema allows
		// the bare aliases "sonnet"/"opus"/"haiku"/"fable" (no "claude-"
		// prefix) for a subagent's model field — none of the HasPrefix
		// checks here matched those, so every subagent requesting the
		// CHEAPEST tier ("haiku") silently fell through to the
		// unmapped-model fallback below and got escalated to the
		// heaviest ModeAuto/unlimited-thinking tier instead, burning
		// 30-60s of extra TTFT and quota on what should've been a quick,
		// cheap subagent call. Handle the bare aliases explicitly.
		model = "gemini-3.8-flash-tiered"
	} else if strings.HasPrefix(model, "claude-sonnet") || strings.HasPrefix(model, "claude-3-5-sonnet") || strings.HasPrefix(model, "claude-3-7-sonnet") || strings.HasPrefix(model, "claude-3-sonnet") || model == "sonnet" {
		model = "gemini-3.8-flash-medium"
	} else if strings.HasPrefix(model, "claude-haiku") || strings.HasPrefix(model, "claude-3-5-haiku") || strings.HasPrefix(model, "claude-3-haiku") || model == "haiku" {
		model = "gemini-2.5-flash"
	} else if model != "" && !strings.HasPrefix(model, "gemini-") {
		// Fallback for unmapped or legacy models (claude-2.1, claude-instant, etc.)
		model = "gemini-3.8-flash-tiered"
	}
	// A Round 15 audit caught that an empty model (the request simply
	// didn't set one) fell into the generic unmapped/legacy fallback
	// above like any other unrecognized name, forcing
	// "gemini-3.8-flash-tiered" — which made
	// ConvertClaudeRequestToGemini's own `if modelName == "" { modelName
	// = tenant.DefaultModel }` fallback (translator.go) permanently dead
	// code, since this function could never actually return "". Leaving
	// model empty here doesn't break the rest of this function: every
	// later check below is an explicit prefix/equality match against a
	// real model name, so an empty model simply falls through to the
	// generic non-tiered budget defaults, same as any other model this
	// function doesn't specifically recognize.

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
		if reqClass == "auxiliary" || isLightweightSubagent {
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
			case "none", "off", "disabled":
				// A Round 15 audit caught that these literal thinking-off
				// values (as opposed to the standard Anthropic
				// thinking.type=="disabled" object, or no effort at all)
				// fell into default: below, which for an Opus-class
				// request selected the HEAVIEST ModeAuto/unlimited-
				// thinking tier — the exact opposite of what was
				// explicitly asked for, costing ~30-40s of extra latency
				// and quota the caller was specifically trying to avoid.
				model = base + "-medium"
				budget = 0
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
		if reqClass == "auxiliary" || isLightweightSubagent {
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
			case "none", "off", "disabled":
				model = base + "-medium"
				budget = 0
			default:
				model = base + "-medium"
				budget = 8192
			}
		}
	} else {
		// Non-tiered models thinking budget
		//
		// An audit caught two related gaps for Haiku (gemini-2.5-flash),
		// whose whole point is a fast, cheap, thinking-free response:
		// (1) the auxiliary/lightweight-subagent branch below ran
		// UNCONDITIONALLY and forced budget=2048 even for Haiku, before
		// the switch's own Haiku zero-budget default case ever got a
		// chance to run; (2) Claude Code's CLAUDE_CODE_EFFORT_LEVEL env
		// var (read client-side, arriving here as output_config.effort)
		// is a GLOBAL default applied to every model including Haiku —
		// "medium"/"low" land straight in the switch's explicit case
		// branches, never reaching the Haiku-specific default either.
		// Only a clear, deliberate high-effort escalation (xhigh/max/
		// ultracode) is still honored for Haiku; that's unlikely to be a
		// mere inherited global default and more likely a genuine ask
		// for Haiku to actually reason hard on this one call.
		if model == "gemini-2.5-flash" && effort != "xhigh" && effort != "max" && effort != "ultracode" {
			budget = 0
		} else if strings.HasSuffix(model, "-thinking") && effort == "" {
			// A Round 15 audit caught that a "-thinking"-suffixed native
			// model (e.g. "claude-opus-4-6-thinking", preserved verbatim
			// by the native-models branch above) never matched any of
			// this switch's -high/-low/-medium suffix checks in its
			// default: case, so with no explicit effort it silently kept
			// budget at this function's generic 8192 initializer instead
			// of the unlimited ModeAuto reasoning its name and purpose
			// call for. Only applies with no explicit effort override —
			// an explicit low/medium/high/xhigh request below still wins.
			budget = -1
		} else if reqClass == "auxiliary" || isLightweightSubagent {
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
			case "none", "off", "disabled":
				budget = 0
			default:
				if strings.HasSuffix(model, "-high") {
					budget = 16384
				} else if strings.HasSuffix(model, "-low") {
					budget = 2048
				} else if strings.HasSuffix(model, "-medium") {
					budget = 8192
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
