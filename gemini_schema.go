// package main provides utility functions for the CLI Proxy API server.
package main

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var gjsonPathKeyReplacer = strings.NewReplacer(".", "\\.", "*", "\\*", "?", "\\?")

// numericEnumFields remembers, per declared tool name, which top-level
// argument fields had a non-string enum (integer/number/boolean) in the
// ORIGINAL Claude tool schema before convertEnumValuesToStrings forced them
// to TYPE_STRING for Gemini — Gemini only allows enum on STRING fields, but
// it then returns the value back as a JSON string ("1") on a function call,
// which Claude Code's own schema validator (still expecting the original
// integer/number/boolean type) rejects. A tool's schema is static for the
// life of the conversation that declared it, so a global cache (same
// pattern as sse.go's toolSignatureCache) needs no request/session
// plumbing through ConvertClaudeRequestToGemini's return values.
var (
	numericEnumFieldsMu sync.Mutex
	numericEnumFields   = make(map[string]map[string]string) // toolName -> fieldName -> original JSON type
)

// openDictArgFields remembers, per declared tool name, which top-level
// argument fields are free-form dicts (originally declared with
// additionalProperties, e.g. Bash's "env" or TaskCreate's "metadata") that
// addEmptySchemaPlaceholder had to inject a "reason" placeholder into — so
// a function call's args can get that nested "env.reason" stripped too, not
// just a top-level "reason" (see syntheticReasonTools in translator.go/
// sse.go, which only ever covered the top level).
var (
	openDictArgFieldsMu sync.Mutex
	openDictArgFields   = make(map[string][]string)
)

func stripNestedOpenDictReason(toolName, argsRaw string) string {
	openDictArgFieldsMu.Lock()
	fields := openDictArgFields[toolName]
	openDictArgFieldsMu.Unlock()
	for _, field := range fields {
		path := field + ".reason"
		if gjson.Get(argsRaw, path).Exists() {
			if cleaned, err := sjson.Delete(argsRaw, path); err == nil {
				argsRaw = cleaned
			}
		}
	}
	return argsRaw
}

// recordNumericEnumFields scans a tool's ORIGINAL (pre-clean) input_schema
// recursively for properties with a non-string enum, so the type can be
// restored on the way back in coerceNumericEnumArgs.
func recordNumericEnumFields(toolName, inputSchemaRaw string) {
	fields := make(map[string]string)
	walkSchemaForNumericEnums(gjson.Get(inputSchemaRaw, "properties"), "", fields)
	numericEnumFieldsMu.Lock()
	defer numericEnumFieldsMu.Unlock()
	if len(fields) == 0 {
		delete(numericEnumFields, toolName)
		return
	}
	numericEnumFields[toolName] = fields
}

func walkSchemaForNumericEnums(props gjson.Result, prefix string, fields map[string]string) {
	if !props.IsObject() {
		return
	}
	props.ForEach(func(key, val gjson.Result) bool {
		path := key.String()
		if prefix != "" {
			path = prefix + "." + key.String()
		}
		t := val.Get("type").String()
		if t == "" && val.Get("enum").IsArray() {
			first := val.Get("enum.0")
			if first.Type == gjson.Number {
				if strings.Contains(first.Raw, ".") {
					t = "number"
				} else {
					t = "integer"
				}
			} else if first.Type == gjson.True || first.Type == gjson.False {
				t = "boolean"
			}
		}
		if val.Get("enum").IsArray() && (t == "integer" || t == "number" || t == "boolean") {
			fields[path] = t
		}
		if val.Get("properties").IsObject() {
			walkSchemaForNumericEnums(val.Get("properties"), path, fields)
		}
		// An array of a numeric/boolean enum (e.g. "ports": {"type":"array",
		// "items":{"type":"integer","enum":[80,443]}}) needs its own marker
		// — each element lives at its own index, not at `path` itself — so
		// coerceNumericEnumArgs knows to walk the array instead of reading
		// `path` as a single scalar.
		if t == "array" {
			items := val.Get("items")
			itemType := items.Get("type").String()
			if itemType == "" && items.Get("enum").IsArray() {
				first := items.Get("enum.0")
				if first.Type == gjson.Number {
					if strings.Contains(first.Raw, ".") {
						itemType = "number"
					} else {
						itemType = "integer"
					}
				} else if first.Type == gjson.True || first.Type == gjson.False {
					itemType = "boolean"
				}
			}
			if items.Get("enum").IsArray() && (itemType == "integer" || itemType == "number" || itemType == "boolean") {
				fields[path] = "array:" + itemType
			}
		}
		return true
	})
}

// coerceNumericEnumArgs restores a function call's string-coerced enum
// fields back to their declared integer/number/boolean type before the
// args are handed to the client as a tool_use block.
func coerceNumericEnumArgs(toolName, argsRaw string) string {
	numericEnumFieldsMu.Lock()
	fields := numericEnumFields[toolName]
	numericEnumFieldsMu.Unlock()
	for field, origType := range fields {
		if itemType, ok := strings.CutPrefix(origType, "array:"); ok {
			arr := gjson.Get(argsRaw, field)
			if !arr.IsArray() {
				continue
			}
			for i, el := range arr.Array() {
				if el.Type != gjson.String {
					continue
				}
				idxPath := fmt.Sprintf("%s.%d", field, i)
				switch itemType {
				case "integer":
					if n, err := strconv.ParseInt(el.String(), 10, 64); err == nil {
						argsRaw, _ = sjson.Set(argsRaw, idxPath, n)
					}
				case "number":
					if f, err := strconv.ParseFloat(el.String(), 64); err == nil {
						argsRaw, _ = sjson.Set(argsRaw, idxPath, f)
					}
				case "boolean":
					if b, err := strconv.ParseBool(el.String()); err == nil {
						argsRaw, _ = sjson.Set(argsRaw, idxPath, b)
					}
				}
			}
			continue
		}
		v := gjson.Get(argsRaw, field)
		if v.Type != gjson.String {
			continue
		}
		switch origType {
		case "integer":
			if n, err := strconv.ParseInt(v.String(), 10, 64); err == nil {
				argsRaw, _ = sjson.Set(argsRaw, field, n)
			}
		case "number":
			if f, err := strconv.ParseFloat(v.String(), 64); err == nil {
				argsRaw, _ = sjson.Set(argsRaw, field, f)
			}
		case "boolean":
			if b, err := strconv.ParseBool(v.String()); err == nil {
				argsRaw, _ = sjson.Set(argsRaw, field, b)
			}
		}
	}
	return argsRaw
}

const placeholderReasonDescription = "Brief explanation of why you are calling this tool"

// CleanJSONSchemaForAntigravity transforms a JSON schema to be compatible with Antigravity API.
// It handles unsupported keywords, type flattening, and schema simplification while preserving
// semantic information as description hints.
func CleanJSONSchemaForAntigravity(toolName, jsonStr string) string {
	cleaned, nestedOpenDictFields := cleanJSONSchema(toolName, jsonStr, true)
	openDictArgFieldsMu.Lock()
	if len(nestedOpenDictFields) > 0 {
		openDictArgFields[toolName] = nestedOpenDictFields
	} else {
		delete(openDictArgFields, toolName)
	}
	openDictArgFieldsMu.Unlock()
	return cleaned
}

// CleanJSONSchemaForGemini transforms a JSON schema to be compatible with Gemini tool calling.
// It removes unsupported keywords and simplifies schemas, without adding empty-schema placeholders.
func CleanJSONSchemaForGemini(jsonStr string) string {
	cleaned, _ := cleanJSONSchema("", jsonStr, false)
	return cleaned
}

// cleanJSONSchema performs the core cleaning operations on the JSON schema.
// Returns the cleaned schema and, when addPlaceholder is set, the top-level
// argument field names addEmptySchemaPlaceholder had to patch because they
// were free-form dicts (see openDictPaths/openDictArgFields above) rather
// than genuinely empty schemas.
func cleanJSONSchema(toolName, jsonStr string, addPlaceholder bool) (string, []string) {
	// A node with additionalProperties set to a SCHEMA (not just `false`) is
	// a free-form dict by design — Bash's "env", TaskCreate's "metadata" —
	// and legitimately has no fixed "properties" of its own. Phase 3 below
	// strips additionalProperties entirely as unsupported, which otherwise
	// leaves nothing behind to distinguish that node from a genuinely empty
	// schema, and Phase 4 would inject a required "reason" placeholder
	// property into it. The model dutifully fills "reason" in, and a tool
	// with a strict handler (additionalProperties: false on ITS end)
	// rejects the call over that one extra field the client-side cleanup
	// elsewhere only ever strips at the top level of the arguments, never
	// inside a nested dict field like this one.
	openDictPaths := make(map[string]bool)
	for _, p := range findPaths(jsonStr, "additionalProperties") {
		if gjson.Get(jsonStr, p).Type != gjson.False {
			openDictPaths[trimSuffix(p, ".additionalProperties")] = true
		}
	}

	// Phase 1: Convert references and inline structures
	jsonStr = convertRefsToHints(jsonStr)
	jsonStr = convertConstToEnum(jsonStr)

	// Phase 2: Flatten complex structures
	jsonStr = mergeAllOf(jsonStr)
	jsonStr = flattenAnyOfOneOf(jsonStr)
	jsonStr = flattenTypeArrays(jsonStr)

	// Record numeric and boolean enums AFTER references are inlined and allOf/anyOf
	// structures are merged, but BEFORE convertEnumValuesToStrings coerces them
	// to string enums for Gemini OpenAPI 3.0. This ensures enums defined via $ref
	// or allOf composition are correctly preserved for coercion upon response.
	if toolName != "" {
		recordNumericEnumFields(toolName, jsonStr)
	}

	jsonStr = convertEnumValuesToStrings(jsonStr)
	jsonStr = addEnumHints(jsonStr)
	jsonStr = addAdditionalPropertiesHints(jsonStr)
	jsonStr = moveConstraintsToDescription(jsonStr)

	// Phase 3: Cleanup
	jsonStr = removeUnsupportedKeywords(jsonStr)
	if !addPlaceholder {
		// Gemini schema cleanup: remove nullable/title and placeholder-only fields.
		jsonStr = removeKeywords(jsonStr, []string{"nullable"})
		jsonStr = removePlaceholderFields(jsonStr)
	}
	jsonStr = cleanupRequiredFields(jsonStr)
	jsonStr = addMissingArrayItems(jsonStr)
	// Phase 4: Add placeholder for empty object schemas (Claude VALIDATED mode requirement)
	var nestedOpenDictFields []string
	if addPlaceholder {
		jsonStr, nestedOpenDictFields = addEmptySchemaPlaceholder(jsonStr, openDictPaths)
	}

	return jsonStr, nestedOpenDictFields
}

// removeKeywords removes all occurrences of specified keywords from the JSON schema.
func removeKeywords(jsonStr string, keywords []string) string {
	deletePaths := make([]string, 0)
	pathsByField := findPathsByFields(jsonStr, keywords)
	for _, key := range keywords {
		for _, p := range pathsByField[key] {
			if isPropertyDefinition(trimSuffix(p, "."+key)) {
				continue
			}
			deletePaths = append(deletePaths, p)
		}
	}
	sortByDepth(deletePaths)
	for _, p := range deletePaths {
		jsonStr, _ = sjson.Delete(jsonStr, p)
	}
	return jsonStr
}

// removePlaceholderFields removes placeholder-only properties ("_" and "reason") and their required entries.
func removePlaceholderFields(jsonStr string) string {
	// Remove "_" placeholder properties.
	paths := findPaths(jsonStr, "_")
	sortByDepth(paths)
	for _, p := range paths {
		if !strings.HasSuffix(p, ".properties._") {
			continue
		}
		jsonStr, _ = sjson.Delete(jsonStr, p)
		parentPath := trimSuffix(p, ".properties._")
		reqPath := joinPath(parentPath, "required")
		req := gjson.Get(jsonStr, reqPath)
		if req.IsArray() {
			var filtered []string
			for _, r := range req.Array() {
				if r.String() != "_" {
					filtered = append(filtered, r.String())
				}
			}
			if len(filtered) == 0 {
				jsonStr, _ = sjson.Delete(jsonStr, reqPath)
			} else {
				jsonStr, _ = sjson.Set(jsonStr, reqPath, filtered)
			}
		}
	}

	// Remove placeholder-only "reason" objects.
	reasonPaths := findPaths(jsonStr, "reason")
	sortByDepth(reasonPaths)
	for _, p := range reasonPaths {
		if !strings.HasSuffix(p, ".properties.reason") {
			continue
		}
		parentPath := trimSuffix(p, ".properties.reason")
		props := gjson.Get(jsonStr, joinPath(parentPath, "properties"))
		if !props.IsObject() || len(props.Map()) != 1 {
			continue
		}
		desc := gjson.Get(jsonStr, p+".description").String()
		if desc != placeholderReasonDescription {
			continue
		}
		jsonStr, _ = sjson.Delete(jsonStr, p)
		reqPath := joinPath(parentPath, "required")
		req := gjson.Get(jsonStr, reqPath)
		if req.IsArray() {
			var filtered []string
			for _, r := range req.Array() {
				if r.String() != "reason" {
					filtered = append(filtered, r.String())
				}
			}
			if len(filtered) == 0 {
				jsonStr, _ = sjson.Delete(jsonStr, reqPath)
			} else {
				jsonStr, _ = sjson.Set(jsonStr, reqPath, filtered)
			}
		}
	}

	return jsonStr
}

// convertRefsToHints converts $ref to description hints (Lazy Hint strategy).
func convertRefsToHints(jsonStr string) string {
	expandedRefPaths := make(map[string]string) // prefixPath -> refVal
	expansionCount := make(map[string]int)      // refVal -> total expansions

	for pass := 0; pass < 5; pass++ {
		paths := findPaths(jsonStr, "$ref")
		if len(paths) == 0 {
			break
		}
		sortByDepth(paths)

		for _, p := range paths {
			refNode := gjson.Get(jsonStr, p)
			if !refNode.Exists() {
				continue
			}
			refVal := refNode.String()
			if refVal == "" {
				continue
			}
			parentPath := trimSuffix(p, ".$ref")

			defName := refVal
			if idx := strings.LastIndex(refVal, "/"); idx >= 0 {
				defName = refVal[idx+1:]
			}

			// True recursion check: is this $ref being evaluated inside an ancestor path
			// that already expanded the exact same refVal?
			isRecursive := false
			for prefix, expandedRef := range expandedRefPaths {
				if expandedRef == refVal {
					if prefix == "" || parentPath == prefix || strings.HasPrefix(parentPath, prefix+".") {
						isRecursive = true
						break
					}
				}
			}
			// Hard safety cap against runaway non-tree schemas
			if expansionCount[refVal] >= 10 {
				isRecursive = true
			}

			if isRecursive {
				replacement := fmt.Sprintf(`{"type":"object","description":"Recursive reference: See %s"}`, defName)
				jsonStr = setRawAt(jsonStr, parentPath, replacement)
				continue
			}

			if def := resolveLocalRef(jsonStr, refVal); def != "" {
				expandedRefPaths[parentPath] = refVal
				expansionCount[refVal]++
				if existingDesc := gjson.Get(jsonStr, joinPath(parentPath, "description")).String(); existingDesc != "" {
					def, _ = sjson.Set(def, "description", existingDesc)
				}
				// If replacing at root, preserve $defs / definitions so subsequent passes can resolve nested refs
				if parentPath == "" {
					if !gjson.Get(def, "$defs").Exists() && gjson.Get(jsonStr, "$defs").Exists() {
						def, _ = sjson.SetRaw(def, "$defs", gjson.Get(jsonStr, "$defs").Raw)
					}
					if !gjson.Get(def, "definitions").Exists() && gjson.Get(jsonStr, "definitions").Exists() {
						def, _ = sjson.SetRaw(def, "definitions", gjson.Get(jsonStr, "definitions").Raw)
					}
				}
				jsonStr = setRawAt(jsonStr, parentPath, def)
				continue
			}

			hint := fmt.Sprintf("See: %s", defName)
			if existing := gjson.Get(jsonStr, descriptionPath(parentPath)).String(); existing != "" {
				hint = fmt.Sprintf("%s (%s)", existing, hint)
			}

			replacement := `{"type":"object","description":""}`
			replacement, _ = sjson.Set(replacement, "description", hint)
			jsonStr = setRawAt(jsonStr, parentPath, replacement)
		}
	}
	return jsonStr
}

// resolveLocalRef looks up a JSON Schema $ref against this same document's
// own "$defs" or "definitions" dictionary (the only two locations this
// function understands — a remote/external $ref URL isn't resolvable here
// and falls back to the lossy hint-stub in convertRefsToHints). Returns ""
// if the ref isn't a local pointer or the named definition doesn't exist.
func resolveLocalRef(jsonStr, refVal string) string {
	const defsPrefix = "#/$defs/"
	const definitionsPrefix = "#/definitions/"
	const componentsSchemasPrefix = "#/components/schemas/"

	var prefix, rootField string
	switch {
	case strings.HasPrefix(refVal, defsPrefix):
		prefix, rootField = defsPrefix, "$defs"
	case strings.HasPrefix(refVal, definitionsPrefix):
		prefix, rootField = definitionsPrefix, "definitions"
	case strings.HasPrefix(refVal, componentsSchemasPrefix):
		prefix, rootField = componentsSchemasPrefix, "components.schemas"
	default:
		return ""
	}

	subPath := refVal[len(prefix):]
	segments := strings.Split(subPath, "/")
	currPath := rootField
	for _, seg := range segments {
		seg = strings.ReplaceAll(seg, "~1", "/")
		seg = strings.ReplaceAll(seg, "~0", "~")
		currPath = currPath + "." + escapeGJSONPathKey(seg)
	}
	if def := gjson.Get(jsonStr, currPath); def.Exists() {
		return def.Raw
	}
	return ""
}

func convertConstToEnum(jsonStr string) string {
	for _, p := range findPaths(jsonStr, "const") {
		val := gjson.Get(jsonStr, p)
		if !val.Exists() {
			continue
		}
		parentPath := trimSuffix(p, ".const")
		if isPropertyDefinition(parentPath) {
			continue
		}
		enumPath := parentPath + ".enum"
		if !gjson.Get(jsonStr, enumPath).Exists() {
			jsonStr, _ = sjson.Set(jsonStr, enumPath, []interface{}{val.Value()})
		}
	}
	return jsonStr
}

// convertEnumValuesToStrings ensures all enum values are strings and the schema type is set to string.
// Gemini API requires enum values to be of type string, not numbers or booleans.
func convertEnumValuesToStrings(jsonStr string) string {
	for _, p := range findPaths(jsonStr, "enum") {
		arr := gjson.Get(jsonStr, p)
		if !arr.IsArray() {
			continue
		}

		var stringVals []string
		for _, item := range arr.Array() {
			stringVals = append(stringVals, item.String())
		}

		// Always update enum values to strings and set type to "string"
		// This ensures compatibility with Antigravity Gemini which only allows enum for STRING type
		jsonStr, _ = sjson.Set(jsonStr, p, stringVals)
		parentPath := trimSuffix(p, ".enum")
		jsonStr, _ = sjson.Set(jsonStr, joinPath(parentPath, "type"), "string")
	}
	return jsonStr
}

func addEnumHints(jsonStr string) string {
	for _, p := range findPaths(jsonStr, "enum") {
		arr := gjson.Get(jsonStr, p)
		if !arr.IsArray() {
			continue
		}
		items := arr.Array()
		if len(items) <= 1 || len(items) > 10 {
			continue
		}

		var vals []string
		for _, item := range items {
			vals = append(vals, item.String())
		}
		jsonStr = appendHint(jsonStr, trimSuffix(p, ".enum"), "Allowed: "+strings.Join(vals, ", "))
	}
	return jsonStr
}

func addAdditionalPropertiesHints(jsonStr string) string {
	for _, p := range findPaths(jsonStr, "additionalProperties") {
		if gjson.Get(jsonStr, p).Type == gjson.False {
			jsonStr = appendHint(jsonStr, trimSuffix(p, ".additionalProperties"), "No extra properties allowed")
		}
	}
	return jsonStr
}

var unsupportedConstraints = []string{
	"minLength", "maxLength", "exclusiveMinimum", "exclusiveMaximum",
	"minimum", "maximum", "multipleOf", // MCP tools with numeric validators commonly set these; Gemini's schema doesn't document support for them
	"pattern", "minItems", "maxItems", "minProperties", "maxProperties", "format",
	"default", "example", "examples", // Claude rejects these in VALIDATED mode
	"prefixItems", "uniqueItems", "contains", "minContains", "maxContains",
	"dependentRequired", "dependentSchemas", "contentEncoding", "contentMediaType",
	"readOnly", "writeOnly", "deprecated",
	"unevaluatedProperties", "unevaluatedItems", "$comment",
	"if", "then", "else", "not",
}

func moveConstraintsToDescription(jsonStr string) string {
	pathsByField := findPathsByFields(jsonStr, unsupportedConstraints)
	for _, key := range unsupportedConstraints {
		for _, p := range pathsByField[key] {
			val := gjson.Get(jsonStr, p)
			if !val.Exists() || val.IsObject() || val.IsArray() {
				continue
			}
			parentPath := trimSuffix(p, "."+key)
			if isPropertyDefinition(parentPath) {
				continue
			}
			jsonStr = appendHint(jsonStr, parentPath, fmt.Sprintf("%s: %s", key, val.String()))
		}
	}
	return jsonStr
}

func mergeAllOf(jsonStr string) string {
	paths := findPaths(jsonStr, "allOf")
	sortByDepth(paths)

	for _, p := range paths {
		allOf := gjson.Get(jsonStr, p)
		if !allOf.IsArray() {
			continue
		}
		parentPath := trimSuffix(p, ".allOf")

		for _, item := range allOf.Array() {
			if props := item.Get("properties"); props.IsObject() {
				if !gjson.Get(jsonStr, joinPath(parentPath, "type")).Exists() {
					jsonStr, _ = sjson.Set(jsonStr, joinPath(parentPath, "type"), "object")
				}
				props.ForEach(func(key, value gjson.Result) bool {
					destPath := joinPath(parentPath, "properties."+escapeGJSONPathKey(key.String()))
					jsonStr, _ = sjson.SetRaw(jsonStr, destPath, value.Raw)
					return true
				})
			}
			if req := item.Get("required"); req.IsArray() {
				reqPath := joinPath(parentPath, "required")
				current := getStrings(jsonStr, reqPath)
				for _, r := range req.Array() {
					if s := r.String(); !contains(current, s) {
						current = append(current, s)
					}
				}
				jsonStr, _ = sjson.Set(jsonStr, reqPath, current)
			}
			for _, kw := range []string{"type", "enum", "format", "description", "items", "nullable"} {
				if v := item.Get(kw); v.Exists() && !gjson.Get(jsonStr, joinPath(parentPath, kw)).Exists() {
					jsonStr, _ = sjson.SetRaw(jsonStr, joinPath(parentPath, kw), v.Raw)
				}
			}
		}
		jsonStr, _ = sjson.Delete(jsonStr, p)
	}
	return jsonStr
}

func flattenAnyOfOneOf(jsonStr string) string {
	for _, key := range []string{"anyOf", "oneOf"} {
		paths := findPaths(jsonStr, key)
		sortByDepth(paths)

		for _, p := range paths {
			arr := gjson.Get(jsonStr, p)
			if !arr.IsArray() || len(arr.Array()) == 0 {
				continue
			}

			parentPath := trimSuffix(p, "."+key)
			parentDesc := gjson.Get(jsonStr, descriptionPath(parentPath)).String()

			items := arr.Array()
			bestIdx, allTypes := selectBest(items)
			selected := items[bestIdx].Raw
			if !gjson.Get(selected, "type").Exists() {
				if gjson.Get(selected, "properties").Exists() {
					selected, _ = sjson.Set(selected, "type", "object")
				} else if gjson.Get(selected, "items").Exists() {
					selected, _ = sjson.Set(selected, "type", "array")
				}
			}

			// If all branches are strings with enums, merge all enum options together!
			allEnums := make([]string, 0)
			allStringEnums := true
			for _, itm := range items {
				t := itm.Get("type").String()
				if t == "null" {
					continue
				}
				if t == "string" && itm.Get("enum").IsArray() {
					for _, ev := range itm.Get("enum").Array() {
						allEnums = append(allEnums, ev.String())
					}
				} else {
					allStringEnums = false
				}
			}
			if allStringEnums && len(allEnums) > 0 {
				enumBytes, _ := json.Marshal(allEnums)
				selected, _ = sjson.SetRaw(selected, "enum", string(enumBytes))
			}

			if parentDesc != "" {
				selected = mergeDescriptionRaw(selected, parentDesc)
			}

			if len(allTypes) > 1 {
				hint := "Accepts: " + strings.Join(allTypes, " | ")
				selected = appendHintRaw(selected, hint)
			}
			if contains(allTypes, "null") {
				// Same reasoning as flattenTypeArrays: collapsing
				// anyOf/oneOf down to its single best branch otherwise
				// threw away the fact that "null" was one of the original
				// options, leaving the model unable to pass an explicit
				// null for a field Gemini's validator now sees as a plain
				// (non-nullable) string/object/etc.
				selected, _ = sjson.Set(selected, "nullable", true)

				// Also strip from parent object's required array if this is a property
				parts := splitGJSONPath(parentPath)
				if len(parts) >= 2 && parts[len(parts)-2] == "properties" {
					fieldName := unescapeGJSONPathKey(parts[len(parts)-1])
					objectPath := strings.Join(parts[:len(parts)-2], ".")
					reqPath := "required"
					if objectPath != "" {
						reqPath = objectPath + ".required"
					}
					if req := gjson.Get(jsonStr, reqPath); req.IsArray() {
						var filtered []string
						for _, r := range req.Array() {
							if r.String() != fieldName {
								filtered = append(filtered, r.String())
							}
						}
						if len(filtered) == 0 {
							jsonStr, _ = sjson.Delete(jsonStr, reqPath)
						} else {
							jsonStr, _ = sjson.Set(jsonStr, reqPath, filtered)
						}
					}
				}
			}

			jsonStr = setRawAt(jsonStr, parentPath, selected)
		}
	}
	return jsonStr
}

func selectBest(items []gjson.Result) (bestIdx int, types []string) {
	bestScore := -1
	for i, item := range items {
		t := item.Get("type").String()
		score := 0

		switch {
		case t == "object" || item.Get("properties").Exists():
			score, t = 3, orDefault(t, "object")
		case t == "array" || item.Get("items").Exists():
			score, t = 2, orDefault(t, "array")
		case t != "" && t != "null":
			score = 1
		default:
			t = orDefault(t, "null")
		}

		if t != "" {
			types = append(types, t)
		}
		if score > bestScore {
			bestScore, bestIdx = score, i
		}
	}
	return
}

func flattenTypeArrays(jsonStr string) string {
	paths := findPaths(jsonStr, "type")
	sortByDepth(paths)

	nullableFields := make(map[string][]string)

	for _, p := range paths {
		res := gjson.Get(jsonStr, p)
		if !res.IsArray() || len(res.Array()) == 0 {
			continue
		}

		hasNull := false
		var nonNullTypes []string
		for _, item := range res.Array() {
			s := item.String()
			if s == "null" {
				hasNull = true
			} else if s != "" {
				nonNullTypes = append(nonNullTypes, s)
			}
		}

		firstType := "string"
		if len(nonNullTypes) > 0 {
			firstType = nonNullTypes[0]
		}

		jsonStr, _ = sjson.Set(jsonStr, p, firstType)

		parentPath := trimSuffix(p, ".type")
		if len(nonNullTypes) > 1 {
			hint := "Accepts: " + strings.Join(nonNullTypes, " | ")
			jsonStr = appendHint(jsonStr, parentPath, hint)
		}

		if hasNull {
			// Antigravity's internal schema (unlike the public Gemini
			// Developer API's stricter OpenAPI 3.0 subset) does have a real
			// "nullable" field — CleanJSONSchemaForGemini strips it back
			// out right after this for that other path, so setting it
			// unconditionally here is inert there and only takes effect on
			// the live Antigravity path. Previously only a "(nullable)"
			// description hint was added and the field dropped from
			// "required" below — workable as long as the model omits the
			// field entirely, but Gemini's own validator rejects an
			// explicit "field": null against a type it never declared as
			// nullable.
			jsonStr, _ = sjson.Set(jsonStr, joinPath(parentPath, "nullable"), true)
			parts := splitGJSONPath(p)
			if len(parts) >= 3 && parts[len(parts)-3] == "properties" {
				fieldNameEscaped := parts[len(parts)-2]
				fieldName := unescapeGJSONPathKey(fieldNameEscaped)
				objectPath := strings.Join(parts[:len(parts)-3], ".")
				nullableFields[objectPath] = append(nullableFields[objectPath], fieldName)

				propPath := joinPath(objectPath, "properties."+fieldNameEscaped)
				jsonStr = appendHint(jsonStr, propPath, "(nullable)")
			}
		}
	}

	for objectPath, fields := range nullableFields {
		reqPath := joinPath(objectPath, "required")
		req := gjson.Get(jsonStr, reqPath)
		if !req.IsArray() {
			continue
		}

		var filtered []string
		for _, r := range req.Array() {
			if !contains(fields, r.String()) {
				filtered = append(filtered, r.String())
			}
		}

		if len(filtered) == 0 {
			jsonStr, _ = sjson.Delete(jsonStr, reqPath)
		} else {
			jsonStr, _ = sjson.Set(jsonStr, reqPath, filtered)
		}
	}
	return jsonStr
}

func removeUnsupportedKeywords(jsonStr string) string {
	keywords := append(unsupportedConstraints,
		"$schema", "$defs", "definitions", "components", "const", "$ref", "$id", "additionalProperties",
		"propertyNames", "patternProperties", // Gemini doesn't support these schema keywords
		"enumTitles", "prefill", // Claude/OpenCode schema metadata fields unsupported by Gemini
		// "title" is commonly auto-generated by Pydantic/TypeScript MCP
		// tool schemas, and unlike a plain JSON Schema validator (which
		// ignores unrecognized annotation keywords), Gemini's proto3-based
		// parameter validator rejects it outright. Was previously only
		// stripped on the CleanJSONSchemaForGemini path (addPlaceholder ==
		// false) — never on CleanJSONSchemaForAntigravity, which is what
		// translator.go actually calls for every tool declaration sent
		// upstream.
		"title",
	)

	deletePaths := make([]string, 0)
	pathsByField := findPathsByFields(jsonStr, keywords)
	for _, key := range keywords {
		for _, p := range pathsByField[key] {
			if isPropertyDefinition(trimSuffix(p, "."+key)) {
				continue
			}
			deletePaths = append(deletePaths, p)
		}
	}
	sortByDepth(deletePaths)
	for _, p := range deletePaths {
		jsonStr, _ = sjson.Delete(jsonStr, p)
	}
	// Remove x-* extension fields (e.g., x-google-enum-descriptions) that are not supported by Gemini API
	jsonStr = removeExtensionFields(jsonStr)
	return jsonStr
}

// removeExtensionFields removes all x-* extension fields from the JSON schema.
// These are OpenAPI/JSON Schema extension fields that Google APIs don't recognize.
func removeExtensionFields(jsonStr string) string {
	var paths []string
	walkForExtensions(gjson.Parse(jsonStr), "", &paths)
	// walkForExtensions returns paths in a way that deeper paths are added before their ancestors
	// when they are not deleted wholesale, but since we skip children of deleted x-* nodes,
	// any collected path is safe to delete. We still use DeleteBytes for efficiency.

	b := []byte(jsonStr)
	for _, p := range paths {
		b, _ = sjson.DeleteBytes(b, p)
	}
	return string(b)
}

func walkForExtensions(value gjson.Result, path string, paths *[]string) {
	if value.IsArray() {
		arr := value.Array()
		for i := len(arr) - 1; i >= 0; i-- {
			walkForExtensions(arr[i], joinPath(path, strconv.Itoa(i)), paths)
		}
		return
	}

	if value.IsObject() {
		value.ForEach(func(key, val gjson.Result) bool {
			keyStr := key.String()
			safeKey := escapeGJSONPathKey(keyStr)
			childPath := joinPath(path, safeKey)

			// If it's an extension field, we delete it and don't need to look at its children.
			if strings.HasPrefix(keyStr, "x-") && !isPropertyDefinition(path) {
				*paths = append(*paths, childPath)
				return true
			}

			walkForExtensions(val, childPath, paths)
			return true
		})
	}
}

func cleanupRequiredFields(jsonStr string) string {
	for _, p := range findPaths(jsonStr, "required") {
		parentPath := trimSuffix(p, ".required")
		propsPath := joinPath(parentPath, "properties")

		req := gjson.Get(jsonStr, p)
		props := gjson.Get(jsonStr, propsPath)
		if !req.IsArray() {
			continue
		}
		if !props.IsObject() {
			jsonStr, _ = sjson.Delete(jsonStr, p)
			continue
		}

		var valid []string
		for _, r := range req.Array() {
			key := r.String()
			if props.Get(escapeGJSONPathKey(key)).Exists() {
				valid = append(valid, key)
			}
		}

		if len(valid) != len(req.Array()) {
			if len(valid) == 0 {
				jsonStr, _ = sjson.Delete(jsonStr, p)
			} else {
				jsonStr, _ = sjson.Set(jsonStr, p, valid)
			}
		}
	}
	return jsonStr
}

// addEmptySchemaPlaceholder adds a placeholder "reason" property to empty object schemas.
// Claude VALIDATED mode requires at least one required property in tool schemas.
// nestedOpenDictFields collects the top-level argument field name (e.g.
// "env") of every NESTED (non-root) object that got the placeholder because
// it was actually a free-form dict (openDictPaths) rather than a genuinely
// empty schema — so the caller can register them for recursive "reason"
// cleanup on the way back from a function call, the same way the top-level
// placeholder is already cleaned up via syntheticReasonTools.
func addEmptySchemaPlaceholder(jsonStr string, openDictPaths map[string]bool) (string, []string) {
	// Find all "type" fields
	paths := findPaths(jsonStr, "type")

	// Process from deepest to shallowest (to handle nested objects properly)
	sortByDepth(paths)

	var nestedOpenDictFields []string
	for _, p := range paths {
		typeVal := gjson.Get(jsonStr, p)
		if typeVal.String() != "object" {
			continue
		}

		// Get the parent path (the object containing "type")
		parentPath := trimSuffix(p, ".type")

		// Check if properties exists and is empty or missing
		propsPath := joinPath(parentPath, "properties")
		propsVal := gjson.Get(jsonStr, propsPath)
		reqPath := joinPath(parentPath, "required")

		needsPlaceholder := false
		if !propsVal.Exists() {
			// No properties field at all
			needsPlaceholder = true
		} else if propsVal.IsObject() && len(propsVal.Map()) == 0 {
			// Empty properties object
			needsPlaceholder = true
		}

		if needsPlaceholder {
			// Add placeholder "reason" property. This is the one case
			// Google's own validator actually rejects without it
			// ("parameters.properties: should be non-empty for OBJECT
			// type") — a genuinely empty properties object.
			reasonPath := joinPath(propsPath, "reason")
			jsonStr, _ = sjson.Set(jsonStr, reasonPath+".type", "string")
			jsonStr, _ = sjson.Set(jsonStr, reasonPath+".description", placeholderReasonDescription)

			// Add to required array
			jsonStr, _ = sjson.Set(jsonStr, reqPath, []string{"reason"})

			if openDictPaths[parentPath] {
				if field, ok := strings.CutPrefix(parentPath, "properties."); ok && !strings.Contains(field, ".") {
					nestedOpenDictFields = append(nestedOpenDictFields, unescapeGJSONPathKey(field))
				}
			}
		}
		// Previously also force-added a bogus "_": {"type":"boolean"}
		// required property to ANY nested object that had real properties
		// but none marked required — not something Google's API has ever
		// been confirmed to reject, and it pollutes the arguments the
		// model actually fills in (and that Claude Code's own tool handler
		// then receives) with a field nothing downstream expects.
	}

	return jsonStr, nestedOpenDictFields
}

// --- Helpers ---

func findPaths(jsonStr, field string) []string {
	var paths []string
	Walk(gjson.Parse(jsonStr), "", field, &paths)
	return paths
}

func findPathsByFields(jsonStr string, fields []string) map[string][]string {
	set := make(map[string]struct{}, len(fields))
	for _, field := range fields {
		set[field] = struct{}{}
	}
	paths := make(map[string][]string, len(set))
	walkForFields(gjson.Parse(jsonStr), "", set, paths)
	return paths
}

func walkForFields(value gjson.Result, path string, fields map[string]struct{}, paths map[string][]string) {
	switch value.Type {
	case gjson.JSON:
		value.ForEach(func(key, val gjson.Result) bool {
			keyStr := key.String()
			safeKey := escapeGJSONPathKey(keyStr)

			var childPath string
			if path == "" {
				childPath = safeKey
			} else {
				childPath = path + "." + safeKey
			}

			if _, ok := fields[keyStr]; ok {
				paths[keyStr] = append(paths[keyStr], childPath)
			}

			walkForFields(val, childPath, fields, paths)
			return true
		})
	case gjson.String, gjson.Number, gjson.True, gjson.False, gjson.Null:
		// Terminal types - no further traversal needed
	}
}

func sortByDepth(paths []string) {
	sort.Slice(paths, func(i, j int) bool { return len(paths[i]) > len(paths[j]) })
}

func trimSuffix(path, suffix string) string {
	if path == strings.TrimPrefix(suffix, ".") {
		return ""
	}
	return strings.TrimSuffix(path, suffix)
}

func joinPath(base, suffix string) string {
	if base == "" {
		return suffix
	}
	return base + "." + suffix
}

func setRawAt(jsonStr, path, value string) string {
	if path == "" {
		return value
	}
	result, _ := sjson.SetRaw(jsonStr, path, value)
	return result
}

func isPropertyDefinition(path string) bool {
	return path == "properties" || strings.HasSuffix(path, ".properties")
}

func descriptionPath(parentPath string) string {
	if parentPath == "" || parentPath == "@this" {
		return "description"
	}
	return parentPath + ".description"
}

func appendHint(jsonStr, parentPath, hint string) string {
	descPath := parentPath + ".description"
	if parentPath == "" || parentPath == "@this" {
		descPath = "description"
	}
	existing := gjson.Get(jsonStr, descPath).String()
	if existing != "" {
		hint = fmt.Sprintf("%s (%s)", existing, hint)
	}
	jsonStr, _ = sjson.Set(jsonStr, descPath, hint)
	return jsonStr
}

func appendHintRaw(jsonRaw, hint string) string {
	existing := gjson.Get(jsonRaw, "description").String()
	if existing != "" {
		hint = fmt.Sprintf("%s (%s)", existing, hint)
	}
	jsonRaw, _ = sjson.Set(jsonRaw, "description", hint)
	return jsonRaw
}

func getStrings(jsonStr, path string) []string {
	var result []string
	if arr := gjson.Get(jsonStr, path); arr.IsArray() {
		for _, r := range arr.Array() {
			result = append(result, r.String())
		}
	}
	return result
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func orDefault(val, def string) string {
	if val == "" {
		return def
	}
	return val
}

func escapeGJSONPathKey(key string) string {
	if !strings.ContainsAny(key, ".*?") {
		return key
	}
	return gjsonPathKeyReplacer.Replace(key)
}

func unescapeGJSONPathKey(key string) string {
	if !strings.Contains(key, "\\") {
		return key
	}
	var b strings.Builder
	b.Grow(len(key))
	for i := 0; i < len(key); i++ {
		if key[i] == '\\' && i+1 < len(key) {
			i++
			b.WriteByte(key[i])
			continue
		}
		b.WriteByte(key[i])
	}
	return b.String()
}

func splitGJSONPath(path string) []string {
	if path == "" {
		return nil
	}

	parts := make([]string, 0, strings.Count(path, ".")+1)
	var b strings.Builder
	b.Grow(len(path))

	for i := 0; i < len(path); i++ {
		c := path[i]
		if c == '\\' && i+1 < len(path) {
			b.WriteByte('\\')
			i++
			b.WriteByte(path[i])
			continue
		}
		if c == '.' {
			parts = append(parts, b.String())
			b.Reset()
			continue
		}
		b.WriteByte(c)
	}
	parts = append(parts, b.String())
	return parts
}

func mergeDescriptionRaw(schemaRaw, parentDesc string) string {
	childDesc := gjson.Get(schemaRaw, "description").String()
	switch {
	case childDesc == "":
		schemaRaw, _ = sjson.Set(schemaRaw, "description", parentDesc)
		return schemaRaw
	case childDesc == parentDesc:
		return schemaRaw
	default:
		combined := fmt.Sprintf("%s (%s)", parentDesc, childDesc)
		schemaRaw, _ = sjson.Set(schemaRaw, "description", combined)
		return schemaRaw
	}
}

func Walk(value gjson.Result, path string, targetKey string, paths *[]string) {
	if value.IsArray() {
		for i, item := range value.Array() {
			childPath := joinPath(path, fmt.Sprintf("%d", i))
			Walk(item, childPath, targetKey, paths)
		}
		return
	}
	if value.IsObject() {
		value.ForEach(func(key, val gjson.Result) bool {
			k := key.String()
			safeKey := escapeGJSONPathKey(k)
			childPath := joinPath(path, safeKey)
			if k == targetKey {
				*paths = append(*paths, childPath)
			}
			Walk(val, childPath, targetKey, paths)
			return true
		})
	}
}

// addMissingArrayItems gives every array schema that declares no "items" an empty one.
//
// Antigravity only takes the protobuf `parameters` field (its executor renames
// `parametersJsonSchema` back to it), and that schema requires items on an array: one
// argument declared as a bare {"type":"array"} fails the whole request with
// "...parameters.properties[x].items: missing field." JSON Schema allows the bare form,
// where it means any element, and MCP servers emit it routinely.
//
// {} keeps that meaning. Upstream CLIProxyAPI fills {"type":"string"} instead, which
// declares an array of objects to be an array of strings; Antigravity's Gemini and Claude
// models were both measured to accept {}.
//
// Runs after flattenTypeArrays, so a ["array","null"] list is a plain "array" by now.
// Request contents are left alone: a function call replayed there can hold
// "type": "array" as data.
func addMissingArrayItems(jsonStr string) string {
	for _, p := range findPaths(jsonStr, "type") {
		if strings.HasPrefix(p, "contents.") || strings.HasPrefix(p, "request.contents.") {
			continue
		}
		parentPath := trimSuffix(p, ".type")
		// A property named "type" is a definition, not the keyword.
		if isPropertyDefinition(parentPath) {
			continue
		}
		declared := gjson.Get(jsonStr, p)
		if declared.Type != gjson.String || !strings.EqualFold(declared.String(), "array") {
			continue
		}
		itemsPath := joinPath(parentPath, "items")
		if gjson.Get(jsonStr, itemsPath).Exists() {
			continue
		}
		jsonStr = setRawAt(jsonStr, itemsPath, "{}")
	}
	return jsonStr
}
