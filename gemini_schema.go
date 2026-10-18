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
		argsRaw = stripReasonAtFieldSpec(argsRaw, "", field)
	}
	return argsRaw
}

// stripReasonAtFieldSpec walks a single openDictArgFields entry against the
// real argument VALUES, deleting a synthetic "reason" placeholder wherever
// the spec resolves. The spec can contain any number of ".items"/".items."
// array markers (addEmptySchemaPlaceholder's own convention for "this
// placeholder sat inside an array's element schema", possibly nested
// several arrays deep — e.g. "matrix.items.items.metadata" for a 2D array
// of arrays of objects with a "metadata" dict). An audit caught the
// original version only ever peeled off ONE such marker before falling
// back to flat string concatenation for everything after it — correct for
// one array dimension, but a second (or deeper) dimension's own "items"
// token then became part of that flat path literally, matching nothing in
// the real data and leaking the placeholder through unstripped.
// pathPrefix is the real gjson/sjson path already resolved so far (starts
// empty at the top level, grows by one concrete array index per
// recursion); field is the remaining, not-yet-resolved tail of the spec.
func stripReasonAtFieldSpec(argsRaw, pathPrefix, field string) string {
	joined := func(key string) string {
		if pathPrefix == "" {
			return key
		}
		if key == "" {
			return pathPrefix
		}
		return pathPrefix + "." + key
	}
	// The spec's leading segment is itself an array marker — either this
	// recursion starts fresh on an "items"-prefixed remainder peeled off
	// by the caller, or (rarer) the whole spec is just "items".
	if field == "items" {
		return stripReasonAcrossArray(argsRaw, joined(""), "")
	}
	if rest, ok := strings.CutPrefix(field, "items."); ok {
		return stripReasonAcrossArray(argsRaw, joined(""), rest)
	}
	if arrayField, ok := strings.CutSuffix(field, ".items"); ok {
		return stripReasonAcrossArray(argsRaw, joined(arrayField), "")
	}
	if idx := strings.Index(field, ".items."); idx != -1 {
		arrayField := field[:idx]
		remainder := field[idx+len(".items."):]
		return stripReasonAcrossArray(argsRaw, joined(arrayField), remainder)
	}
	// No more array markers — a plain nested-object path, delete "reason"
	// directly under it.
	path := joined(field) + ".reason"
	if gjson.Get(argsRaw, path).Exists() {
		if cleaned, err := sjson.Delete(argsRaw, path); err == nil {
			argsRaw = cleaned
		}
	}
	return argsRaw
}

// stripReasonAcrossArray iterates every element of the array at arrayPath,
// recursing into stripReasonAtFieldSpec for each one with remainder as the
// still-unresolved tail — remainder may itself contain further array
// markers for additional dimensions.
func stripReasonAcrossArray(argsRaw, arrayPath, remainder string) string {
	arr := gjson.Get(argsRaw, arrayPath)
	if !arr.IsArray() {
		return argsRaw
	}
	n := len(arr.Array())
	for i := n - 1; i >= 0; i-- {
		elemPrefix := fmt.Sprintf("%s.%d", arrayPath, i)
		if remainder == "" {
			path := elemPrefix + ".reason"
			if gjson.Get(argsRaw, path).Exists() {
				if cleaned, err := sjson.Delete(argsRaw, path); err == nil {
					argsRaw = cleaned
				}
			}
			continue
		}
		argsRaw = stripReasonAtFieldSpec(argsRaw, elemPrefix, remainder)
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
			// An audit caught that an array of OBJECTS (items.properties,
			// not items.enum directly) was never recursed into at all —
			// e.g. "ports": {"type":"array","items":{"type":"object",
			// "properties":{"port":{"type":"integer","enum":[80,443]}}}}.
			// convertEnumValuesToStrings still stringifies "port"'s enum
			// for Gemini, but nothing here ever recorded it, so Gemini's
			// string response ("80") was never coerced back to an integer,
			// failing Claude Code's own schema validation on the result.
			// Record these with an "arrayObject:<subfield>:<type>" marker
			// (distinct from the flat "array:<type>" scalar case above,
			// which has no sub-field to name) since the array's length
			// isn't known until the actual tool-call arguments arrive.
			if items.Get("properties").IsObject() {
				items.Get("properties").ForEach(func(subKey, subVal gjson.Result) bool {
					subT := subVal.Get("type").String()
					if subT == "" && subVal.Get("enum").IsArray() {
						first := subVal.Get("enum.0")
						if first.Type == gjson.Number {
							if strings.Contains(first.Raw, ".") {
								subT = "number"
							} else {
								subT = "integer"
							}
						} else if first.Type == gjson.True || first.Type == gjson.False {
							subT = "boolean"
						}
					}
					if subVal.Get("enum").IsArray() && (subT == "integer" || subT == "number" || subT == "boolean") {
						fields[path+"[]."+subKey.String()] = "arrayObject:" + subT
					}
					return true
				})
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
		if itemType, ok := strings.CutPrefix(origType, "arrayObject:"); ok {
			arrayPath, subKey, found := strings.Cut(field, "[].")
			if !found {
				continue
			}
			arr := gjson.Get(argsRaw, arrayPath)
			if !arr.IsArray() {
				continue
			}
			for i, el := range arr.Array() {
				v := el.Get(subKey)
				if v.Type != gjson.String {
					continue
				}
				idxPath := fmt.Sprintf("%s.%d.%s", arrayPath, i, subKey)
				switch itemType {
				case "integer":
					if n, err := strconv.ParseInt(v.String(), 10, 64); err == nil {
						argsRaw, _ = sjson.Set(argsRaw, idxPath, n)
					}
				case "number":
					if f, err := strconv.ParseFloat(v.String(), 64); err == nil {
						argsRaw, _ = sjson.Set(argsRaw, idxPath, f)
					}
				case "boolean":
					if b, err := strconv.ParseBool(v.String()); err == nil {
						argsRaw, _ = sjson.Set(argsRaw, idxPath, b)
					}
				}
			}
			continue
		}
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
			dictPath := trimSuffix(p, ".additionalProperties")
			openDictPaths[dictPath] = true
			// An audit caught that a free-form dict declared only as
			// {"additionalProperties": {"type": "string"}} — no "type"
			// of its own, a common way to write this in Pydantic/
			// TypeScript-generated schemas — loses its additionalProperties
			// in Phase 3 below with nothing left behind to mark it as an
			// object. Phase 4's addEmptySchemaPlaceholder only ever looks
			// at paths where "type" ALREADY exists (findPaths(jsonStr,
			// "type")), so this node was skipped entirely and reached
			// Gemini as a bare "{}" — "Schema cannot be empty". Force
			// "type":"object" now so Phase 4 sees it.
			if !gjson.Get(jsonStr, joinPath(dictPath, "type")).Exists() {
				jsonStr, _ = sjson.Set(jsonStr, joinPath(dictPath, "type"), "object")
			}
		}
	}

	// An audit caught that a root schema which implies "object" by having
	// "properties" (valid per JSON Schema Draft 7/2020-12 — explicit
	// "type" is optional when properties are present) reached Gemini
	// without one: translator.go's own empty-schema placeholder check
	// only fires when "properties" is EMPTY, so a non-empty-but-untyped
	// root schema slipped through as parameters: {"properties": ...}
	// with no "type": "OBJECT", which Gemini's upstream rejects with a
	// 400. Force it explicitly here, at the root, before anything else runs.
	if gjson.Get(jsonStr, "properties").Exists() && !gjson.Get(jsonStr, "type").Exists() {
		jsonStr, _ = sjson.Set(jsonStr, "type", "object")
	}

	// Phase 1: Convert references and inline structures
	jsonStr = convertRefsToHints(jsonStr)
	jsonStr = convertConstToEnum(jsonStr)
	jsonStr = convertPrefixItemsToItems(jsonStr)

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
	jsonStr = addMissingPropertyTypes(jsonStr)
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
		// An audit caught that leaving "type" unset here (it was only ever
		// filled in later by convertEnumValuesToStrings, which runs AFTER
		// flattenAnyOfOneOf) made every const-derived branch of an
		// anyOf/oneOf look untyped (type=="") to flattenAnyOfOneOf's
		// allStringEnums/selectBest logic — so instead of merging all the
		// branches' enum values together, it silently kept only the first
		// branch and mislabeled the field "nullable" on top of that. Set
		// the type immediately so these branches are recognized as string
		// enums right away.
		typePath := parentPath + ".type"
		if gjson.Get(jsonStr, typePath).Exists() {
			continue
		}
		// An audit caught that only val.Type == gjson.String was ever
		// typed here — a numeric or boolean const (e.g. status: const 1)
		// stayed untyped, so flattenAnyOfOneOf's allStringEnums check
		// (which only recognizes t == "string") treated that branch as
		// not-a-string-enum and refused to merge it with sibling string
		// consts, and selectBest's own type inference fell back to
		// "null" for it, falsely marking the whole field nullable while
		// silently discarding the numeric/boolean branch.
		var inferredType string
		switch val.Type {
		case gjson.String:
			inferredType = "string"
		case gjson.Number:
			inferredType = "number"
			if val.Num == float64(int64(val.Num)) {
				inferredType = "integer"
			}
		case gjson.True, gjson.False:
			inferredType = "boolean"
		}
		if inferredType != "" {
			jsonStr, _ = sjson.Set(jsonStr, typePath, inferredType)
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
	"dependencies", "dependentRequired", "dependentSchemas", "contentEncoding", "contentMediaType",
	// An audit caught patternProperties was only ever in
	// removeUnsupportedKeywords' DELETE list, never in this
	// extract-to-description list — a free-dict schema validating its
	// keys by regex (e.g. "^[A-Z_]+$": {...}) lost that constraint
	// completely with no trace, instead of degrading to a description
	// hint like every other unsupported keyword here.
	"patternProperties",
	"readOnly", "writeOnly", "deprecated",
	"unevaluatedProperties", "unevaluatedItems", "$comment",
	"if", "then", "else", "not",
}

func moveConstraintsToDescription(jsonStr string) string {
	pathsByField := findPathsByFields(jsonStr, unsupportedConstraints)
	for _, key := range unsupportedConstraints {
		for _, p := range pathsByField[key] {
			val := gjson.Get(jsonStr, p)
			if !val.Exists() {
				continue
			}
			parentPath := trimSuffix(p, "."+key)
			if isPropertyDefinition(parentPath) {
				continue
			}
			if val.IsObject() || val.IsArray() {
				// An audit caught that object/array-valued constraints —
				// dependentRequired/dependencies/dependentSchemas describe
				// real structural rules ("if credit_card is set,
				// billing_address becomes required"), not just a scalar
				// limit like minLength — were skipped here entirely, with
				// no hint of any kind. removeUnsupportedKeywords then
				// deletes the keyword outright later in the pipeline
				// (every key in unsupportedConstraints is also in that
				// delete list), so the rule vanished completely instead
				// of degrading to a plain-text description, the same way
				// every scalar constraint here already does. Use the
				// compact JSON form as the hint text — lossy (the model
				// can't enforce it structurally either way, same
				// limitation as every other constraint this function
				// handles), but strictly better than total silence.
				jsonStr = appendHint(jsonStr, parentPath, fmt.Sprintf("%s: %s", key, val.Raw))
				continue
			}
			jsonStr = appendHint(jsonStr, parentPath, fmt.Sprintf("%s: %s", key, val.String()))
		}
	}
	return jsonStr
}

// mergeAllOfPropertiesInto merges one allOf branch's "properties" object
// into whatever already lives at propsDestPath (another branch's own
// "properties", possibly already merged from earlier branches).
//
// An audit caught that the collision path below only ever copied a
// colliding key's ATTRIBUTES (type, description, other scalar keywords) and
// explicitly skipped "properties" once any properties object already
// existed at that destination — so {allOf: [{properties:{config:
// {properties:{a}}}}, {properties:{config:{properties:{b}}}}]} merged
// "config" once (branch 1, with only "a") and then, on branch 2, saw
// destPath+".properties" already existed and silently dropped "b" entirely
// instead of merging it alongside "a". Recursing into "properties" instead
// of treating it as an opaque atomic value fixes this at any nesting depth.
func mergeAllOfPropertiesInto(jsonStr, propsDestPath string, props gjson.Result) string {
	props.ForEach(func(key, value gjson.Result) bool {
		destPath := joinPath(propsDestPath, escapeGJSONPathKey(key.String()))
		if !gjson.Get(jsonStr, destPath).Exists() {
			jsonStr, _ = sjson.SetRaw(jsonStr, destPath, value.Raw)
			return true
		}
		// Deep merge collision: preserve existing type if incoming lacks one, merge description hints
		existingType := gjson.Get(jsonStr, destPath+".type").String()
		incomingType := value.Get("type").String()
		if existingType == "" && incomingType != "" {
			jsonStr, _ = sjson.Set(jsonStr, destPath+".type", incomingType)
		}
		existingDesc := gjson.Get(jsonStr, destPath+".description").String()
		incomingDesc := value.Get("description").String()
		if incomingDesc != "" && incomingDesc != existingDesc {
			jsonStr = appendHint(jsonStr, destPath, incomingDesc)
		}
		value.ForEach(func(attrKey, attrVal gjson.Result) bool {
			k := attrKey.String()
			if k == "description" || k == "type" {
				return true
			}
			if k == "properties" && attrVal.IsObject() {
				jsonStr = mergeAllOfPropertiesInto(jsonStr, destPath+".properties", attrVal)
				return true
			}
			if k == "items" && attrVal.IsObject() {
				// An audit caught this fell through to the generic
				// "only set if destPath.k doesn't exist yet" rule below
				// too, same as "required" above was fixed for — when two
				// allOf branches both constrain an array's element
				// schema (e.g. one branch adds a "name" field to items,
				// another adds "id"), only the FIRST branch's "items" was
				// ever kept, destroying the second branch's own
				// constraints on the array's elements.
				itemsDestPath := destPath + ".items"
				if !gjson.Get(jsonStr, itemsDestPath).Exists() {
					jsonStr, _ = sjson.SetRaw(jsonStr, itemsDestPath, attrVal.Raw)
					return true
				}
				if itemsProps := attrVal.Get("properties"); itemsProps.IsObject() {
					jsonStr = mergeAllOfPropertiesInto(jsonStr, itemsDestPath+".properties", itemsProps)
				}
				if itemsReq := attrVal.Get("required"); itemsReq.IsArray() {
					reqPath := itemsDestPath + ".required"
					seen := make(map[string]bool)
					var merged []string
					if existing := gjson.Get(jsonStr, reqPath); existing.IsArray() {
						for _, r := range existing.Array() {
							if !seen[r.String()] {
								seen[r.String()] = true
								merged = append(merged, r.String())
							}
						}
					}
					for _, r := range itemsReq.Array() {
						if !seen[r.String()] {
							seen[r.String()] = true
							merged = append(merged, r.String())
						}
					}
					jsonStr, _ = sjson.Set(jsonStr, reqPath, merged)
				}
				return true
			}
			if k == "required" && attrVal.IsArray() {
				// An audit caught that this fell through to the generic
				// "only set if destPath.k doesn't exist yet" rule below —
				// so when two allOf branches collide on the same nested
				// object (e.g. both declare a "user" sub-object) and each
				// requires a DIFFERENT field of it (branch 1 requires
				// "name", branch 2 requires "email"), branch 2's required
				// array was silently dropped entirely instead of unioned,
				// since branch 1's array already existed at destPath.
				reqPath := destPath + ".required"
				existing := gjson.Get(jsonStr, reqPath)
				seen := make(map[string]bool)
				var merged []string
				if existing.IsArray() {
					for _, r := range existing.Array() {
						if !seen[r.String()] {
							seen[r.String()] = true
							merged = append(merged, r.String())
						}
					}
				}
				for _, r := range attrVal.Array() {
					if !seen[r.String()] {
						seen[r.String()] = true
						merged = append(merged, r.String())
					}
				}
				jsonStr, _ = sjson.Set(jsonStr, reqPath, merged)
				return true
			}
			if !gjson.Get(jsonStr, destPath+"."+k).Exists() {
				jsonStr, _ = sjson.SetRaw(jsonStr, destPath+"."+k, attrVal.Raw)
			}
			return true
		})
		return true
	})
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
				jsonStr = mergeAllOfPropertiesInto(jsonStr, joinPath(parentPath, "properties"), props)
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
			// anyOf/oneOf added — an audit caught that an allOf branch like
			// {base}, {oneOf: [variantA, variantB]} (a standard Draft 7/
			// 2020-12 pattern) had its oneOf silently dropped: this loop
			// only ever copied the fixed keyword list below, and the
			// unconditional sjson.Delete(jsonStr, p) after this loop then
			// deleted the WHOLE allOf array, variants and all, before
			// flattenAnyOfOneOf (which runs immediately after mergeAllOf)
			// ever got a chance to see them.
			for _, kw := range []string{"type", "enum", "format", "description", "items", "nullable", "anyOf", "oneOf"} {
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

			// An audit caught that a discriminated union of multiple
			// object branches (e.g. action:"create" with a "data" param
			// vs. action:"delete" with an "id" param) lost every branch
			// but the first: selectBest scores every object branch
			// identically (3), and "score > bestScore" never lets a tie
			// win, so bestIdx always stayed 0 — branch 1..N's properties
			// and required fields were discarded entirely, making the
			// model unable to ever invoke any action but the first one.
			// When more than one branch is a genuine object, merge all of
			// their properties together instead of picking just one.
			objectBranches := make([]gjson.Result, 0, len(items))
			for _, itm := range items {
				t := itm.Get("type").String()
				if t == "object" || (t == "" && itm.Get("properties").Exists()) {
					objectBranches = append(objectBranches, itm)
				}
			}
			if len(objectBranches) > 1 {
				selected = mergeObjectBranches(objectBranches)
			}

			// If all non-null branches declare an enum, merge all their
			// options together! An audit caught that this only ever
			// recognized t == "string" — but convertConstToEnum now
			// infers "integer"/"number"/"boolean" too for non-string
			// consts (so sibling branches of a numeric/boolean union
			// aren't left untyped and mis-flagged nullable), and those
			// typed branches then failed this check and got silently
			// dropped instead of merged. Since convertEnumValuesToStrings
			// (which runs right after this) stringifies every enum value
			// and forces type:"string" regardless, merging enums across
			// differently-typed scalar branches is safe here — Gemini
			// only ever sees the final string-typed result either way.
			allEnums := make([]string, 0)
			allStringEnums := true
			for _, itm := range items {
				t := itm.Get("type").String()
				if t == "null" {
					continue
				}
				if itm.Get("enum").IsArray() {
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

			// An audit caught that multiple object branches (now merged
			// above into one combined schema, not picked from) each
			// contributed their own "object" entry to allTypes, producing
			// a useless "Accepts: object | object" hint instead of
			// nothing — dedupe before building the hint.
			dedupedTypes := make([]string, 0, len(allTypes))
			seenType := make(map[string]bool)
			for _, t := range allTypes {
				if !seenType[t] {
					seenType[t] = true
					dedupedTypes = append(dedupedTypes, t)
				}
			}
			if len(dedupedTypes) > 1 {
				hint := "Accepts: " + strings.Join(dedupedTypes, " | ")
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

			// Was: jsonStr = setRawAt(jsonStr, parentPath, selected) — a
			// full replace of whatever lives at parentPath. An audit caught
			// that this destroys any SIBLING property declared alongside
			// anyOf/oneOf at the same level (e.g. a base "action" field
			// plus variant-specific ones per branch: {"type":"object",
			// "properties":{"action":...}, "oneOf":[...]}) — when parentPath
			// is the schema root (""), setRawAt's own "path == \"\" means
			// replace the whole document" special case makes this total.
			// Merge selected's top-level keys into whatever's already at
			// parentPath instead, only filling in keys not already present
			// (properties merge key-by-key, same spirit as mergeAllOf
			// above), then delete the now-redundant anyOf/oneOf array.
			selParsed := gjson.Parse(selected)
			selParsed.ForEach(func(k, v gjson.Result) bool {
				skey := k.String()
				switch skey {
				case "properties":
					if !v.IsObject() {
						return true
					}
					v.ForEach(func(pk, pv gjson.Result) bool {
						propDest := joinPath(parentPath, "properties."+escapeGJSONPathKey(pk.String()))
						if !gjson.Get(jsonStr, propDest).Exists() {
							jsonStr, _ = sjson.SetRaw(jsonStr, propDest, pv.Raw)
						}
						return true
					})
					return true
				case "description":
					// Was "only write if parentPath has none yet" — an audit
					// caught that `selected`'s own description at this point
					// already IS the fully-combined one (parentDesc merged
					// in via mergeDescriptionRaw above, plus the "Accepts:
					// type1 | type2" hint for a multi-type union) — if
					// parentPath already had ITS OWN plain description
					// before this function ran (the common case: that's
					// exactly where parentDesc came from), the generic
					// "skip if exists" rule above threw the combined
					// version away and left the plain original in place,
					// silently dropping the Accepts hint and the variant's
					// own description. This one key must always overwrite.
					destPath := joinPath(parentPath, "description")
					jsonStr, _ = sjson.Set(jsonStr, destPath, v.String())
					return true
				case "required":
					if !v.IsArray() {
						return true
					}
					// Union with whatever's already required at parentPath
					// instead of "skip if parentPath has any required array
					// at all" — an audit caught that the generic rule
					// dropped the selected branch's own required fields
					// (e.g. "value") whenever the parent already required
					// something else (e.g. "action"), even though both are
					// genuinely required once merged.
					destPath := joinPath(parentPath, "required")
					existing := gjson.Get(jsonStr, destPath)
					seen := make(map[string]bool)
					var merged []string
					if existing.IsArray() {
						for _, r := range existing.Array() {
							if !seen[r.String()] {
								seen[r.String()] = true
								merged = append(merged, r.String())
							}
						}
					}
					for _, r := range v.Array() {
						if !seen[r.String()] {
							seen[r.String()] = true
							merged = append(merged, r.String())
						}
					}
					jsonStr, _ = sjson.Set(jsonStr, destPath, merged)
					return true
				}
				destPath := joinPath(parentPath, skey)
				if !gjson.Get(jsonStr, destPath).Exists() {
					jsonStr, _ = sjson.SetRaw(jsonStr, destPath, v.Raw)
				}
				return true
			})
			jsonStr, _ = sjson.Delete(jsonStr, p)
		}
	}
	return jsonStr
}

// mergeObjectBranches combines multiple object-typed anyOf/oneOf branches
// (a discriminated union, e.g. action:"create" vs action:"delete") into one
// schema instead of the caller picking just one and discarding the rest.
// Properties are unioned (first branch to declare a given key wins on a
// genuine name clash); a field present in more than one branch as a scalar
// `const` (the discriminator itself) is merged into a combined `enum`
// instead of keeping only the first branch's single value; `required` is
// the INTERSECTION across branches, since a field only some branches
// require isn't safely mandatory once every branch's parameters are merged
// into one object the model can fill in for any action.
func mergeObjectBranches(branches []gjson.Result) string {
	type propEntry struct {
		raw          string
		constValues  []string
		isConstField bool
		// An audit caught that merging a shared nested object property's
		// own "required" list (right below) took the UNION of every
		// branch's nested required fields, unlike the top-level required
		// merge further down (which correctly takes the INTERSECTION via
		// requiredSets) — a shared "config" sub-object where branch
		// "create" needs config.create_token and branch "delete" needs
		// config.delete_token ended up requiring BOTH simultaneously,
		// making every call fail validation regardless of which action
		// was chosen. Track each branch's own nested required list here
		// so the intersection can be computed the same way as top-level,
		// instead of blindly accumulating.
		nestedRequiredSets [][]string
	}
	order := make([]string, 0)
	props := make(map[string]*propEntry)
	var requiredSets [][]string

	for _, branch := range branches {
		var reqList []string
		if req := branch.Get("required"); req.IsArray() {
			for _, r := range req.Array() {
				reqList = append(reqList, r.String())
			}
		}
		requiredSets = append(requiredSets, reqList)

		if p := branch.Get("properties"); p.IsObject() {
			p.ForEach(func(pk, pv gjson.Result) bool {
				key := pk.String()
				entry, exists := props[key]
				if !exists {
					entry = &propEntry{raw: pv.Raw}
					props[key] = entry
					order = append(order, key)
				} else if pv.Get("properties").IsObject() && gjson.Parse(entry.raw).Get("properties").IsObject() {
					// An audit caught that a non-discriminator shared
					// property — a nested object like "config" that
					// different branches declare with DIFFERENT sub-fields
					// for their own action — used first-branch-wins here:
					// only the FIRST branch's version of "config" was ever
					// kept, silently erasing every other branch's own
					// sub-properties for it. Recursively merge, the same
					// way mergeAllOfPropertiesInto already does for allOf
					// collisions.
					entry.raw = mergeAllOfPropertiesInto(entry.raw, "properties", pv.Get("properties"))
				}
				// Record EVERY branch's own nested required list for this
				// property (not just the merge case) — an audit caught the
				// old code UNIONed these as it went, requiring a sub-field
				// from EVERY branch simultaneously (e.g. config.create_token
				// AND config.delete_token both mandatory) instead of only
				// the sub-fields common to ALL branches that declare this
				// property, mirroring the (correct) top-level
				// requiredSets/intersection logic below.
				if pv.Get("properties").IsObject() {
					var nestedReq []string
					if newReq := pv.Get("required"); newReq.IsArray() {
						for _, r := range newReq.Array() {
							nestedReq = append(nestedReq, r.String())
						}
					}
					entry.nestedRequiredSets = append(entry.nestedRequiredSets, nestedReq)
				}
				if c := pv.Get("const"); c.Exists() {
					entry.isConstField = true
					entry.constValues = append(entry.constValues, c.String())
				} else if en := pv.Get("enum"); en.IsArray() && len(en.Array()) >= 1 {
					// An audit caught that JSON Schema Draft-04/07 commonly
					// expresses a discriminated-union tag as a single-
					// element enum ("enum":["delete"]) rather than "const" —
					// semantically identical, but this only ever checked
					// "const", so a second branch's enum-tagged discriminator
					// value was silently dropped (its properties never even
					// got merged, since the "already exists" check above
					// skips straight past a repeated property key).
					//
					// A follow-up audit caught the len==1 restriction itself:
					// a single branch can validly cover MULTIPLE discriminator
					// values at once ("enum":["create","insert"]), and that
					// whole branch's values were dropped entirely rather than
					// just its first one — accumulate every element, not only
					// a lone one.
					entry.isConstField = true
					for _, v := range en.Array() {
						entry.constValues = append(entry.constValues, v.String())
					}
				}
				return true
			})
		}
	}

	mergedProps := "{}"
	for _, key := range order {
		entry := props[key]
		raw := entry.raw
		if entry.isConstField && len(entry.constValues) > 1 {
			seen := make(map[string]bool)
			var vals []string
			for _, v := range entry.constValues {
				if !seen[v] {
					seen[v] = true
					vals = append(vals, v)
				}
			}
			if len(vals) > 1 {
				enumBytes, _ := json.Marshal(vals)
				raw, _ = sjson.Delete(raw, "const")
				raw, _ = sjson.SetRaw(raw, "enum", string(enumBytes))
				if !gjson.Get(raw, "type").Exists() {
					raw, _ = sjson.Set(raw, "type", "string")
				}
			}
		}
		if len(entry.nestedRequiredSets) > 0 {
			nestedCounts := make(map[string]int)
			for _, set := range entry.nestedRequiredSets {
				seen := make(map[string]bool)
				for _, f := range set {
					if !seen[f] {
						seen[f] = true
						nestedCounts[f]++
					}
				}
			}
			var nestedMerged []string
			for f, n := range nestedCounts {
				if n == len(entry.nestedRequiredSets) {
					nestedMerged = append(nestedMerged, f)
				}
			}
			sort.Strings(nestedMerged)
			if len(nestedMerged) > 0 {
				raw, _ = sjson.Set(raw, "required", nestedMerged)
			} else {
				raw, _ = sjson.Delete(raw, "required")
			}
		}
		mergedProps, _ = sjson.SetRaw(mergedProps, escapeGJSONPathKey(key), raw)
	}

	var mergedRequired []string
	if len(requiredSets) > 0 {
		counts := make(map[string]int)
		for _, set := range requiredSets {
			seen := make(map[string]bool)
			for _, f := range set {
				if !seen[f] {
					seen[f] = true
					counts[f]++
				}
			}
		}
		for _, key := range order {
			if counts[key] == len(requiredSets) {
				mergedRequired = append(mergedRequired, key)
			}
		}
	}

	selected := `{"type":"object"}`
	selected, _ = sjson.SetRaw(selected, "properties", mergedProps)
	if len(mergedRequired) > 0 {
		reqBytes, _ := json.Marshal(mergedRequired)
		selected, _ = sjson.SetRaw(selected, "required", string(reqBytes))
	}
	return selected
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
		case t == "" && item.Get("enum").IsArray() && len(item.Get("enum").Array()) > 0:
			// An audit caught that a branch using "enum" without an
			// explicit "type" (valid JSON Schema — the type is implied by
			// the enum's own values) fell straight into the default
			// "null" case below, same as a genuinely untyped/empty
			// branch. flattenAnyOfOneOf then saw "null" in the collected
			// types and both marked the field nullable AND stripped it
			// from "required" — turning an enum-typed required parameter
			// into an optional, falsely-nullable one. Infer the type from
			// the enum's own first value instead, the same way
			// convertConstToEnum already infers a type for "const".
			switch item.Get("enum.0").Type {
			case gjson.Number:
				score, t = 1, "number"
			case gjson.True, gjson.False:
				score, t = 1, "boolean"
			default:
				score, t = 1, "string"
			}
		default:
			// An audit caught that a branch with NO type signal at all —
			// not even an enum, e.g. just {"description": "any valid
			// filter"} — fell into this same default as a genuinely
			// explicit {"type":"null"} branch (t is already "null" going
			// in for that case, so orDefault leaves it alone either way).
			// flattenAnyOfOneOf can't tell "unknown type" from "explicitly
			// nullable" apart afterward, and treats both as proof the
			// field can be null — wrongly marking a required parameter
			// nullable and stripping it from "required" just because
			// its type happened to be unknown, not because anything ever
			// said it could be absent. Only a genuinely explicit null
			// should reach that path; an unknown type falls back to
			// OpenAPI's base scalar type instead.
			if t == "" {
				t = "string"
			}
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
		if res.Type == gjson.String && res.String() == "null" {
			// A bare scalar "type": "null" (not inside a ["string","null"]
			// array, which the array branch below already handles) slipped
			// through unchanged — Gemini's protobuf Schema only recognizes
			// STRING/NUMBER/INTEGER/BOOLEAN/ARRAY/OBJECT, never "null", and
			// rejects it outright. Same "string + nullable:true" fallback
			// the array case below uses when every type option was null.
			parentPath := trimSuffix(p, ".type")
			jsonStr, _ = sjson.Set(jsonStr, p, "string")
			jsonStr, _ = sjson.Set(jsonStr, joinPath(parentPath, "nullable"), true)
			continue
		}
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
		// An audit caught that picking purely by array POSITION broke a
		// type array like ["string","object"] whose schema ALSO declares
		// "properties" — Gemini's own validator rejects a string-typed
		// field carrying "properties"/"items", a combination that only
		// makes sense for the object/array member of the union. Prefer
		// whichever declared type actually matches the schema's own
		// structural keywords over the union's declaration order.
		parentPathPeek := trimSuffix(p, ".type")
		if gjson.Get(jsonStr, joinPath(parentPathPeek, "properties")).IsObject() && contains(nonNullTypes, "object") {
			firstType = "object"
		} else if gjson.Get(jsonStr, joinPath(parentPathPeek, "items")).Exists() && contains(nonNullTypes, "array") {
			firstType = "array"
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
		// An audit caught "id" (the bare Draft 4 schema-identification
		// keyword, e.g. {"id": "http://...", "properties": {...}}) was
		// missing alongside its Draft 7+ successor "$id" — removeKeywords'
		// isPropertyDefinition guard already protects a genuine tool
		// parameter literally named "id" (properties.id), so adding the
		// bare keyword here is safe.
		"$schema", "$defs", "definitions", "components", "const", "$ref", "$id", "id", "additionalProperties",
		"propertyNames", "patternProperties", // Gemini doesn't support these schema keywords
		"enumTitles", "prefill", // Claude/OpenCode schema metadata fields unsupported by Gemini
		// additionalItems (Draft 7, paired with tuple-style "items": [...])
		// and contentSchema (Draft 2020-12, paired with contentMediaType/
		// contentEncoding — already stripped above) slip through the same
		// way: Gemini's protobuf parser rejects the unknown field outright.
		// $anchor/$dynamicAnchor/$dynamicRef/$vocabulary are Draft 2020-12
		// schema-identification keywords Gemini has no concept of either.
		"additionalItems", "contentSchema", "$anchor", "$dynamicAnchor", "$dynamicRef", "$vocabulary",
		// An audit caught these OpenAPI 3.0 annotation keywords were never
		// stripped at all: "discriminator" (polymorphic oneOf/anyOf
		// schemas), "xml" and "externalDocs" (OpenAPI doc-generation
		// metadata). Any tool schema carrying one passes through to
		// Gemini's protobuf parser unchanged and gets rejected outright
		// with "unknown field <name> in Schema".
		"discriminator", "xml", "externalDocs",
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

			if field, ok := strings.CutPrefix(parentPath, "properties."); ok {
				argPath := strings.ReplaceAll(field, ".properties.", ".")
				// An audit caught that unescaping here, so a property
				// name with a literal dot (e.g. "app.env") round-tripped
				// back to that literal dot instead of staying as the
				// escaped "app\.env" segment it already was, broke
				// stripNestedOpenDictReason downstream: it builds a
				// gjson/sjson path by plain string concatenation
				// (field+".reason"), and both libraries treat an
				// unescaped "." as a path separator — "app.env.reason"
				// parses as three nested levels (app -> env -> reason),
				// not "the literal key app.env, then reason", so the
				// synthetic placeholder never matched and leaked through
				// to the client. Keep it in its already-escaped form;
				// nothing downstream needs the literal (unescaped) name.
				nestedOpenDictFields = append(nestedOpenDictFields, argPath)
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
	sort.Slice(paths, func(i, j int) bool {
		depthI := strings.Count(paths[i], ".")
		depthJ := strings.Count(paths[j], ".")
		if depthI != depthJ {
			return depthI > depthJ
		}
		return len(paths[i]) > len(paths[j])
	})
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

// convertPrefixItemsToItems rewrites Draft 2020-12 tuple validation
// ("prefixItems": [schemaA, schemaB, ...]) into an "items" schema before
// "prefixItems" is stripped as an unsupported keyword later in the pipeline.
//
// An audit caught that without this, a tuple's element schemas were simply
// discarded: "prefixItems" is in unsupportedConstraints (deleted outright by
// Phase 3), "items" never existed on these schemas to begin with, and
// addMissingArrayItems then fills that gap with a bare {} — collapsing a
// described tuple into a completely untyped array. Gemini's protobuf Schema
// has no tuple concept either, so this can only approximate: a single
// prefixItems entry becomes that entry's own schema; several become a oneOf
// of them (flattenAnyOfOneOf, which runs right after this in Phase 2,
// collapses that down further) — far closer to the original intent than an
// empty items.
func convertPrefixItemsToItems(jsonStr string) string {
	for _, p := range findPaths(jsonStr, "prefixItems") {
		arr := gjson.Get(jsonStr, p)
		if !arr.IsArray() || len(arr.Array()) == 0 {
			continue
		}
		parentPath := trimSuffix(p, ".prefixItems")
		itemsPath := joinPath(parentPath, "items")
		if gjson.Get(jsonStr, itemsPath).Exists() {
			continue
		}
		items := arr.Array()
		if len(items) == 1 {
			jsonStr, _ = sjson.SetRaw(jsonStr, itemsPath, items[0].Raw)
			continue
		}
		oneOf := "["
		for i, it := range items {
			if i > 0 {
				oneOf += ","
			}
			oneOf += it.Raw
		}
		oneOf += "]"
		jsonStr, _ = sjson.SetRaw(jsonStr, itemsPath, `{"oneOf":`+oneOf+`}`)
	}
	return jsonStr
}

// addMissingPropertyTypes gives every object property that declares no type
// of its own a default one.
//
// An audit caught that addEmptySchemaPlaceholder only ever looks at paths
// where "type" ALREADY equals "object" (it walks findPaths(jsonStr, "type")),
// so a property with no "type" field at all — e.g. {"description": "any
// payload"}, which Pydantic/TypeScript generators emit routinely for a
// deliberately open-ended field — was never visited by anything and reached
// Gemini's protobuf Schema with no type, which rejects it outright
// ("parameters.properties[x].type: missing field"). By this point in the
// pipeline (after mergeAllOf/flattenAnyOfOneOf/convertConstToEnum),
// "properties"/"items"/"enum"/a surviving composition keyword already means
// a type is implied; only genuinely bare property schemas are affected.
func addMissingPropertyTypes(jsonStr string) string {
	for _, p := range findPaths(jsonStr, "properties") {
		propsVal := gjson.Get(jsonStr, p)
		if !propsVal.IsObject() {
			continue
		}
		propsVal.ForEach(func(k, v gjson.Result) bool {
			if !v.IsObject() {
				return true
			}
			if v.Get("type").Exists() || v.Get("enum").Exists() ||
				v.Get("anyOf").Exists() || v.Get("oneOf").Exists() || v.Get("allOf").Exists() {
				return true
			}
			propPath := joinPath(p, escapeGJSONPathKey(k.String()))
			switch {
			case v.Get("properties").Exists():
				jsonStr, _ = sjson.Set(jsonStr, joinPath(propPath, "type"), "object")
			case v.Get("items").Exists():
				jsonStr, _ = sjson.Set(jsonStr, joinPath(propPath, "type"), "array")
			default:
				jsonStr, _ = sjson.Set(jsonStr, joinPath(propPath, "type"), "string")
			}
			return true
		})
	}
	// An audit caught that the loop above only ever fixed a missing
	// "type" on a PROPERTY found inside some "properties" object — it
	// never visited an array's "items" schema itself (only items'S OWN
	// nested properties, one level deeper). An array declared as
	// items: {"description": "..."} or items: {"properties": {...}}
	// with no explicit "type" on items left items untyped, which
	// Gemini's protobuf schema validator rejects outright
	// ("items.type: missing field"). Scan "items" paths the same way.
	for _, p := range findPaths(jsonStr, "items") {
		v := gjson.Get(jsonStr, p)
		if !v.IsObject() || v.Get("type").Exists() || v.Get("enum").Exists() ||
			v.Get("anyOf").Exists() || v.Get("oneOf").Exists() || v.Get("allOf").Exists() {
			continue
		}
		if v.Get("properties").Exists() {
			jsonStr, _ = sjson.Set(jsonStr, joinPath(p, "type"), "object")
		} else {
			jsonStr, _ = sjson.Set(jsonStr, joinPath(p, "type"), "string")
		}
	}
	return jsonStr
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
		itemsRes := gjson.Get(jsonStr, itemsPath)
		if itemsRes.Exists() {
			if itemsRes.IsArray() {
				// Draft 7 tuple syntax: "items": [{"type": "string"}, {"type": "number"}]
				// OpenAPI 3.0 requires "items" to be a single Schema object!
				firstItem := "{}"
				if len(itemsRes.Array()) > 0 {
					firstItem = itemsRes.Array()[0].Raw
				}
				jsonStr = setRawAt(jsonStr, itemsPath, firstItem)
				jsonStr = appendHint(jsonStr, parentPath, "Tuple items: "+itemsRes.Raw)
			}
			continue
		}
		jsonStr = setRawAt(jsonStr, itemsPath, "{}")
	}
	return jsonStr
}
