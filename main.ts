import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 80;
const MAX_CLIENT_BODY_BYTES = 8 * 1024 * 1024;
const MAX_UPSTREAM_BODY_BYTES = 4 * 1024 * 1024;
const MAX_ATTEMPTS_PER_PROMPT = 5;
const ATTEMPT_TIMEOUT_MS = 5400;
const RETRY_BACKOFF_MS = 200;
const MAX_TRAIL_ITEMS = 8;
const MAX_CODE_LENGTH = 64;

const SCHEMA_PREFIX = 'free-model-aggregator-02/';
const OPENROUTER_PUBLIC_URL = 'https://openrouter.ai/api/v1';
const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';
const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
const FAILURES_HEADER = 'x-makefun-upstream-failures';

const LEGAL_OPENAI_FINISH = new Set(['stop', 'length', 'tool_calls', 'function_call', 'content_filter']);
const TOKEN_PATTERN = /(?:^|[^A-Z0-9])([A-Z]{3,12}-\d{4})(?![A-Z0-9])/;

type JsonObject = Record<string, unknown>;
type WireKind = 'openai' | 'anthropic';

interface ModelSpec {
  id: string;
  provider: string;
  displayName: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
  order: number;
}

interface PromptRequirement {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number | null;
}

interface PromptManifestRow {
  id: string;
  api: WireKind;
  stream: boolean;
  weight: 1 | 2;
  requires: PromptRequirement;
}

interface ParsedBrief {
  models: ModelSpec[];
  prompts: PromptManifestRow[];
  error: string | null;
}

interface CatalogDocument {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: 'openai';
    models: Array<{
      id: string;
      free: boolean;
      context_length: number | null;
    }>;
  }>;
}

interface FailureTrailItem {
  status: number;
  code: string;
  model: string | null;
}

interface ParsedToolCall {
  id: string;
  name: string;
  argumentsText: string;
  argumentsObject: JsonObject;
}

interface NormalizedCompletion {
  id: string;
  text: string;
  textChunks: string[];
  toolCalls: ParsedToolCall[];
  finishReason: string;
  inputTokens: number;
  outputTokens: number;
}

interface ModelRuntimeHealth {
  dead: boolean;
  noJson: boolean;
  noTools: boolean;
  strikes: number;
  wins: number;
}

function isObject(value: unknown): value is JsonObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function findSchemaObjects(source: string): JsonObject[] {
  const docs: JsonObject[] = [];
  for (let start = source.indexOf('{'); start !== -1; start = source.indexOf('{', start + 1)) {
    const end = findClosingBrace(source, start);
    if (end === -1) continue;
    try {
      const parsed: unknown = JSON.parse(source.slice(start, end + 1));
      if (isObject(parsed) && typeof parsed.schema === 'string') {
        docs.push(parsed);
      }
    } catch {
      // Ignore non-JSON spans in prose.
    }
  }
  return docs;
}

function findClosingBrace(source: string, openIndex: number): number {
  let depth = 0;
  let insideString = false;
  for (let i = openIndex; i < source.length; i++) {
    const ch = source[i];
    if (insideString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') insideString = false;
      continue;
    }
    if (ch === '"') insideString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function parseBriefFile(filePath: string): ParsedBrief {
  if (filePath === '') {
    return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  }
  let rawText: string;
  try {
    rawText = readFileSync(filePath, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `failed to read brief at ${filePath}: ${(err as Error).message}` };
  }

  const docs = findSchemaObjects(rawText).filter((d) => asString(d.schema).includes(SCHEMA_PREFIX));
  let rawModels: unknown[] | null = null;
  let rawPrompts: unknown[] | null = null;

  for (const doc of docs) {
    if (rawModels === null) {
      if (Array.isArray(doc.models)) rawModels = doc.models;
      else if (Array.isArray(doc.ratings)) rawModels = doc.ratings;
    }
    if (rawPrompts === null) {
      if (Array.isArray(doc.prompts)) rawPrompts = doc.prompts;
      else if (Array.isArray(doc.manifest)) rawPrompts = doc.manifest;
    }
  }

  if (rawModels === null) {
    return { models: [], prompts: [], error: `no models document found in ${filePath}` };
  }

  const models: ModelSpec[] = [];
  for (let i = 0; i < rawModels.length; i++) {
    const item = rawModels[i];
    if (!isObject(item)) continue;
    const baseId = asString(item.id) !== '' ? asString(item.id) : asString(item.model);
    if (baseId === '') continue;
    const providerRaw = asString(item.provider) !== '' ? asString(item.provider) : asString(item.wire);
    const provider = providerRaw !== '' ? providerRaw : 'openrouter';
    const dialId = provider === 'openrouter' && !baseId.toLowerCase().startsWith('openrouter/') ? `openrouter/${baseId}` : baseId;
    models.push({
      id: dialId,
      provider,
      displayName: asString(item.display_name) !== '' ? asString(item.display_name) : baseId,
      ratingMilli: asFiniteNumber(item.rating_milli) ?? 0,
      contextLength: asFiniteNumber(item.context_length),
      tools: item.tools === true,
      vision: item.vision === true,
      json: item.json === true,
      free: item.free !== false,
      order: i,
    });
  }

  const prompts: PromptManifestRow[] = [];
  if (rawPrompts !== null) {
    for (const item of rawPrompts) {
      if (!isObject(item)) continue;
      const id = asString(item.id);
      if (id === '') continue;
      const reqObj = isObject(item.requires) ? item.requires : {};
      prompts.push({
        id,
        api: item.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: item.stream === true,
        weight: item.weight === 2 ? 2 : 1,
        requires: {
          tools: reqObj.tools === true,
          vision: reqObj.vision === true,
          json: reqObj.json === true,
          minContext: asFiniteNumber(reqObj.min_context),
        },
      });
    }
  }

  return { models, prompts, error: rawPrompts === null ? `no prompts document found in ${filePath}` : null };
}

function buildCatalogDocument(brief: ParsedBrief): CatalogDocument {
  const seen = new Set<string>();
  const models: Array<{ id: string; free: boolean; context_length: number | null }> = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({
      id: model.id,
      free: true,
      context_length: model.contextLength,
    });
  }
  if (models.length === 0) {
    return { providers: [] };
  }
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_URL,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}

function modelSatisfies(model: ModelSpec, req: PromptRequirement): boolean {
  if (!model.free) return false;
  if (req.tools && !model.tools) return false;
  if (req.vision && !model.vision) return false;
  if (req.json && !model.json) return false;
  if (req.minContext !== null && (model.contextLength === null || model.contextLength < req.minContext)) return false;
  return true;
}

function inspectBodyRequirements(wire: WireKind, body: JsonObject): PromptRequirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const msg of messages) {
    if (!isObject(msg) || !Array.isArray(msg.content)) continue;
    for (const part of msg.content) {
      if (isObject(part) && (part.type === 'image' || part.type === 'image_url')) {
        vision = true;
      }
    }
  }
  const hasTools = Array.isArray(body.tools) && body.tools.length > 0;
  let needsJson = false;
  if (wire === 'openai' && isObject(body.response_format)) {
    const fmt = body.response_format.type;
    needsJson = fmt === 'json_object' || fmt === 'json_schema';
  }
  return { tools: hasTools, vision, json: needsJson, minContext: null };
}

function mergeRequirements(a: PromptRequirement, b: PromptRequirement): PromptRequirement {
  return {
    tools: a.tools || b.tools,
    vision: a.vision || b.vision,
    json: a.json || b.json,
    minContext: a.minContext !== null && b.minContext !== null ? Math.max(a.minContext, b.minContext) : (a.minContext ?? b.minContext),
  };
}

function declaredToolsFromBody(wire: WireKind, body: JsonObject): Set<string> | null {
  if (!Array.isArray(body.tools) || body.tools.length === 0) return null;
  const names = new Set<string>();
  for (const raw of body.tools) {
    if (!isObject(raw)) continue;
    if (wire === 'openai' && isObject(raw.function) && typeof raw.function.name === 'string' && raw.function.name !== '') {
      names.add(raw.function.name);
    } else if (wire === 'anthropic' && typeof raw.name === 'string' && raw.name !== '') {
      names.add(raw.name);
    }
  }
  return names;
}

function extractTextBlocks(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const item of content) {
    if (isObject(item) && item.type === 'text' && typeof item.text === 'string') {
      parts.push(item.text);
    }
  }
  return parts.join('\n');
}

function extractExpectedToken(body: JsonObject): string | null {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (!isObject(msg) || msg.role !== 'user') continue;
    const text = extractTextBlocks(msg.content);
    const match = TOKEN_PATTERN.exec(text);
    if (match !== null && typeof match[1] === 'string') {
      return match[1];
    }
  }
  return null;
}

function clampMaxTokens(raw: unknown): number {
  if (typeof raw === 'number' && Number.isFinite(raw)) {
    const rounded = Math.floor(raw);
    if (rounded >= 1 && rounded <= 8192) return rounded;
    if (rounded > 8192) return 8192;
  }
  return 512;
}

function buildOpenAIUpstreamBody(clientBody: JsonObject, modelId: string, stream: boolean): JsonObject {
  const out: JsonObject = {};
  for (const [k, v] of Object.entries(clientBody)) {
    if (k === 'store') continue;
    out[k] = v;
  }
  out.model = modelId;
  if (out.max_tokens === undefined && out.max_completion_tokens === undefined) {
    out.max_tokens = 512;
  } else if (out.max_tokens !== undefined) {
    out.max_tokens = clampMaxTokens(out.max_tokens);
  } else {
    out.max_completion_tokens = clampMaxTokens(out.max_completion_tokens);
  }
  if (stream) {
    out.stream = true;
    const existingOpts = isObject(out.stream_options) ? out.stream_options : {};
    out.stream_options = { ...existingOpts, include_usage: true };
  } else {
    delete out.stream;
    delete out.stream_options;
  }
  return out;
}

function toolResultAsString(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = extractTextBlocks(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function anthropicBlockToOpenAIPart(block: JsonObject): JsonObject | null {
  if (block.type === 'text' && typeof block.text === 'string') {
    return { type: 'text', text: block.text };
  }
  if (block.type === 'image' && isObject(block.source)) {
    const src = block.source;
    if (src.type === 'base64' && typeof src.data === 'string') {
      const mime = typeof src.media_type === 'string' && src.media_type !== '' ? src.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${mime};base64,${src.data}` } };
    }
    if (src.type === 'url' && typeof src.url === 'string') {
      return { type: 'image_url', image_url: { url: src.url } };
    }
  }
  return null;
}

function buildOpenAIFromAnthropic(clientBody: JsonObject, modelId: string, stream: boolean): JsonObject {
  const messages: JsonObject[] = [];
  const sysText = extractTextBlocks(clientBody.system);
  if (sysText !== '') {
    messages.push({ role: 'system', content: sysText });
  }

  const rawMessages = Array.isArray(clientBody.messages) ? clientBody.messages : [];
  for (const entry of rawMessages) {
    if (!isObject(entry)) continue;
    const isAssistant = entry.role === 'assistant';
    const content = entry.content;

    if (typeof content === 'string') {
      messages.push({ role: isAssistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;

    if (isAssistant) {
      const toolCalls: JsonObject[] = [];
      for (const block of content) {
        if (!isObject(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' && block.id !== '' ? block.id : `call_${toolCalls.length}`,
          type: 'function',
          function: {
            name: asString(block.name),
            arguments: JSON.stringify(isObject(block.input) ? block.input : {}),
          },
        });
      }
      const text = extractTextBlocks(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : (toolCalls.length > 0 ? null : ''),
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }

    const userParts: JsonObject[] = [];
    let hasImagePart = false;
    for (const block of content) {
      if (!isObject(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: asString(block.tool_use_id),
          content: toolResultAsString(block.content),
        });
        continue;
      }
      const converted = anthropicBlockToOpenAIPart(block);
      if (converted !== null) {
        if (converted.type === 'image_url') hasImagePart = true;
        userParts.push(converted);
      }
    }

    if (userParts.length > 0) {
      if (hasImagePart) {
        messages.push({ role: 'user', content: userParts });
      } else {
        const joined = userParts.map((p) => asString(p.text)).join('\n');
        messages.push({ role: 'user', content: joined });
      }
    }
  }

  const out: JsonObject = {
    model: modelId,
    messages,
    max_tokens: clampMaxTokens(clientBody.max_tokens),
  };

  if (typeof clientBody.temperature === 'number' && Number.isFinite(clientBody.temperature)) {
    out.temperature = clientBody.temperature;
  }
  if (typeof clientBody.top_p === 'number' && Number.isFinite(clientBody.top_p)) {
    out.top_p = clientBody.top_p;
  }
  if (Array.isArray(clientBody.stop_sequences) && clientBody.stop_sequences.length > 0) {
    out.stop = clientBody.stop_sequences;
  }

  if (Array.isArray(clientBody.tools) && clientBody.tools.length > 0) {
    const mappedTools: JsonObject[] = [];
    for (const rawTool of clientBody.tools) {
      if (!isObject(rawTool) || typeof rawTool.name !== 'string' || rawTool.name === '') continue;
      mappedTools.push({
        type: 'function',
        function: {
          name: rawTool.name,
          description: asString(rawTool.description),
          parameters: isObject(rawTool.input_schema) ? rawTool.input_schema : { type: 'object', properties: {} },
        },
      });
    }
    if (mappedTools.length > 0) {
      out.tools = mappedTools;
    }
  }

  if (isObject(clientBody.tool_choice)) {
    const kind = clientBody.tool_choice.type;
    if (kind === 'auto') out.tool_choice = 'auto';
    else if (kind === 'any') out.tool_choice = 'required';
    else if (kind === 'none') out.tool_choice = 'none';
    else if (kind === 'tool' && typeof clientBody.tool_choice.name === 'string' && clientBody.tool_choice.name !== '') {
      out.tool_choice = { type: 'function', function: { name: clientBody.tool_choice.name } };
    }
  }

  if (stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }

  return out;
}

function parseArgumentsObject(raw: unknown): { text: string; obj: JsonObject } {
  if (isObject(raw)) {
    return { text: JSON.stringify(raw), obj: raw };
  }
  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    if (trimmed !== '') {
      try {
        const parsed: unknown = JSON.parse(trimmed);
        if (isObject(parsed)) {
          return { text: raw, obj: parsed };
        }
      } catch {
        // Fall through to empty object.
      }
    }
  }
  return { text: '{}', obj: {} };
}

function normalizeOpenAIFinish(rawFinish: unknown, hasToolCalls: boolean): string {
  if (typeof rawFinish === 'string' && LEGAL_OPENAI_FINISH.has(rawFinish)) {
    return rawFinish;
  }
  return hasToolCalls ? 'tool_calls' : 'stop';
}

function toAnthropicStopReason(openAIFinish: string): string {
  if (openAIFinish === 'length') return 'max_tokens';
  if (openAIFinish === 'tool_calls' || openAIFinish === 'function_call') return 'tool_use';
  if (openAIFinish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function normalizeTokenCounts(rawUsage: unknown): { inputTokens: number; outputTokens: number } {
  const u = isObject(rawUsage) ? rawUsage : {};
  const rawIn = asFiniteNumber(u.prompt_tokens) ?? asFiniteNumber(u.input_tokens) ?? 0;
  const rawOut = asFiniteNumber(u.completion_tokens) ?? asFiniteNumber(u.output_tokens) ?? 1;
  const inputTokens = Math.max(0, Math.floor(rawIn));
  const outputTokens = Math.max(1, Math.floor(rawOut));
  return { inputTokens, outputTokens };
}

function extractErrorCode(rawText: string): string {
  if (rawText === '') return '';
  try {
    const parsed: unknown = JSON.parse(rawText);
    if (isObject(parsed) && isObject(parsed.error)) {
      const err = parsed.error;
      if (typeof err.code === 'string' && err.code !== '') return err.code.slice(0, MAX_CODE_LENGTH);
      if (typeof err.code === 'number' && Number.isFinite(err.code)) return String(err.code).slice(0, MAX_CODE_LENGTH);
      if (typeof err.type === 'string' && err.type !== '') return err.type.slice(0, MAX_CODE_LENGTH);
    }
  } catch {
    // Plain text error body.
  }
  return '';
}

function parseJsonCompletion(rawText: string): { ok: true; completion: NormalizedCompletion } | { ok: false; inBandStatus: number | null; inBandCode: string } {
  let doc: unknown;
  try {
    doc = JSON.parse(rawText);
  } catch {
    return { ok: false, inBandStatus: null, inBandCode: '' };
  }
  if (!isObject(doc)) {
    return { ok: false, inBandStatus: null, inBandCode: '' };
  }
  if (isObject(doc.error) && !Array.isArray(doc.choices)) {
    const codeVal = doc.error.code;
    const status = typeof codeVal === 'number' && Number.isSafeInteger(codeVal) && codeVal >= 400 && codeVal <= 599 ? codeVal : 502;
    const codeStr = typeof codeVal === 'string' ? codeVal : typeof codeVal === 'number' ? String(codeVal) : asString(doc.error.type);
    return { ok: false, inBandStatus: status, inBandCode: codeStr.slice(0, MAX_CODE_LENGTH) };
  }
  const choices = Array.isArray(doc.choices) ? doc.choices : null;
  if (choices === null || choices.length === 0 || !isObject(choices[0])) {
    return { ok: false, inBandStatus: null, inBandCode: '' };
  }
  const choice = choices[0];
  const message = isObject(choice.message) ? choice.message : null;
  if (message === null) {
    return { ok: false, inBandStatus: null, inBandCode: '' };
  }
  let text = '';
  if (typeof message.content === 'string') {
    text = message.content;
  } else if (Array.isArray(message.content)) {
    text = extractTextBlocks(message.content);
  }
  const toolCalls: ParsedToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (let i = 0; i < message.tool_calls.length; i++) {
      const tc = message.tool_calls[i];
      if (!isObject(tc) || !isObject(tc.function)) continue;
      const name = asString(tc.function.name);
      if (name === '') continue;
      const args = parseArgumentsObject(tc.function.arguments);
      toolCalls.push({
        id: typeof tc.id === 'string' && tc.id !== '' ? tc.id : `call_${i}`,
        name,
        argumentsText: args.text,
        argumentsObject: args.obj,
      });
    }
  }
  const finishReason = normalizeOpenAIFinish(choice.finish_reason, toolCalls.length > 0);
  const { inputTokens, outputTokens } = normalizeTokenCounts(doc.usage);
  const id = typeof doc.id === 'string' && doc.id !== '' ? doc.id : 'chatcmpl-agg';
  return {
    ok: true,
    completion: {
      id,
      text,
      textChunks: text !== '' ? [text] : [],
      toolCalls,
      finishReason,
      inputTokens,
      outputTokens,
    },
  };
}

function parseSseCompletion(rawText: string): { ok: true; completion: NormalizedCompletion } | { ok: false; inBandStatus: number | null; inBandCode: string } {
  const normalized = rawText.replace(/\r\n/g, '\n');
  let sawAnyChunk = false;
  let id = 'chatcmpl-stream';
  const textChunks: string[] = [];
  let rawFinish: unknown = null;
  let usageObj: unknown = null;
  const toolBuilders = new Map<number, { id: string; name: string; args: string }>();

  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    const dataLines: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trimStart();
      if (trimmed.startsWith('data:')) {
        dataLines.push(trimmed.slice(5).trimStart());
      }
    }
    if (dataLines.length === 0) continue;
    const payload = dataLines.join('\n').trim();
    if (payload === '' || payload === '[DONE]') continue;

    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isObject(chunk)) continue;
    if (isObject(chunk.error) && !Array.isArray(chunk.choices)) {
      const codeVal = chunk.error.code;
      const status = typeof codeVal === 'number' && Number.isSafeInteger(codeVal) && codeVal >= 400 && codeVal <= 599 ? codeVal : 502;
      const codeStr = typeof codeVal === 'string' ? codeVal : typeof codeVal === 'number' ? String(codeVal) : asString(chunk.error.type);
      return { ok: false, inBandStatus: status, inBandCode: codeStr.slice(0, MAX_CODE_LENGTH) };
    }

    sawAnyChunk = true;
    if (typeof chunk.id === 'string' && chunk.id !== '') id = chunk.id;
    if (isObject(chunk.usage)) usageObj = chunk.usage;

    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0 || !isObject(choices[0])) continue;
    const choice = choices[0];
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
      rawFinish = choice.finish_reason;
    }
    const delta = isObject(choice.delta) ? choice.delta : (isObject(choice.message) ? choice.message : null);
    if (delta === null) continue;

    if (typeof delta.content === 'string' && delta.content !== '') {
      textChunks.push(delta.content);
    } else if (Array.isArray(delta.content)) {
      const piece = extractTextBlocks(delta.content);
      if (piece !== '') textChunks.push(piece);
    }

    if (Array.isArray(delta.tool_calls)) {
      for (const tc of delta.tool_calls) {
        if (!isObject(tc)) continue;
        const idx = typeof tc.index === 'number' && Number.isSafeInteger(tc.index) ? tc.index : toolBuilders.size;
        let entry = toolBuilders.get(idx);
        if (entry === undefined) {
          entry = { id: '', name: '', args: '' };
          toolBuilders.set(idx, entry);
        }
        if (typeof tc.id === 'string' && tc.id !== '') entry.id = tc.id;
        if (isObject(tc.function)) {
          if (typeof tc.function.name === 'string' && tc.function.name !== '') entry.name += tc.function.name;
          if (typeof tc.function.arguments === 'string') entry.args += tc.function.arguments;
        }
      }
    }
  }

  if (!sawAnyChunk) {
    return { ok: false, inBandStatus: null, inBandCode: '' };
  }

  const toolCalls: ParsedToolCall[] = [];
  for (const [idx, builder] of [...toolBuilders.entries()].sort((a, b) => a[0] - b[0])) {
    if (builder.name === '') continue;
    const parsedArgs = parseArgumentsObject(builder.args);
    toolCalls.push({
      id: builder.id !== '' ? builder.id : `call_${idx}`,
      name: builder.name,
      argumentsText: parsedArgs.text,
      argumentsObject: parsedArgs.obj,
    });
  }

  const text = textChunks.join('');
  const finishReason = normalizeOpenAIFinish(rawFinish, toolCalls.length > 0);
  const { inputTokens, outputTokens } = normalizeTokenCounts(usageObj);

  return {
    ok: true,
    completion: {
      id,
      text,
      textChunks,
      toolCalls,
      finishReason,
      inputTokens,
      outputTokens,
    },
  };
}

function validateCompletion(
  completion: NormalizedCompletion,
  expectedToken: string | null,
  declaredTools: Set<string> | null,
): boolean {
  if (declaredTools !== null) {
    for (const tc of completion.toolCalls) {
      if (!declaredTools.has(tc.name)) return false;
    }
  }
  if (expectedToken !== null) {
    return completion.text.includes(expectedToken);
  }
  return completion.text.length > 0 || completion.toolCalls.length > 0;
}

function renderOpenAIJson(c: NormalizedCompletion, modelId: string): JsonObject {
  const message: JsonObject = {
    role: 'assistant',
    content: c.text !== '' || c.toolCalls.length === 0 ? c.text : null,
  };
  if (c.toolCalls.length > 0) {
    message.tool_calls = c.toolCalls.map((tc) => ({
      id: tc.id,
      type: 'function',
      function: {
        name: tc.name,
        arguments: tc.argumentsText,
      },
    }));
  }
  return {
    id: c.id,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model: modelId,
    choices: [
      {
        index: 0,
        message,
        finish_reason: c.finishReason,
      },
    ],
    usage: {
      prompt_tokens: c.inputTokens,
      completion_tokens: c.outputTokens,
      total_tokens: c.inputTokens + c.outputTokens,
    },
  };
}

function renderAnthropicJson(c: NormalizedCompletion, modelId: string): JsonObject {
  const content: JsonObject[] = [];
  if (c.text !== '' || c.toolCalls.length === 0) {
    content.push({ type: 'text', text: c.text });
  }
  for (const tc of c.toolCalls) {
    content.push({
      type: 'tool_use',
      id: tc.id.startsWith('toolu_') ? tc.id : `toolu_${tc.id}`,
      name: tc.name,
      input: tc.argumentsObject,
    });
  }
  return {
    id: c.id.startsWith('msg_') ? c.id : `msg_${c.id}`,
    type: 'message',
    role: 'assistant',
    model: modelId,
    content,
    stop_reason: toAnthropicStopReason(c.finishReason),
    stop_sequence: null,
    usage: {
      input_tokens: c.inputTokens,
      output_tokens: c.outputTokens,
    },
  };
}

function sseFrame(event: string | null, data: unknown): string {
  const prefix = event !== null ? `event: ${event}\n` : '';
  const payload = typeof data === 'string' ? data : JSON.stringify(data);
  return `${prefix}data: ${payload}\n\n`;
}

function renderOpenAISse(c: NormalizedCompletion, modelId: string): string {
  const frames: string[] = [];
  frames.push(
    sseFrame(null, {
      id: c.id,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model: modelId,
      choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
    }),
  );

  for (const piece of c.textChunks) {
    if (piece === '') continue;
    frames.push(
      sseFrame(null, {
        id: c.id,
        object: 'chat.completion.chunk',
        created: FIXED_OPENAI_CREATED,
        model: modelId,
        choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
      }),
    );
  }

  for (let i = 0; i < c.toolCalls.length; i++) {
    const tc = c.toolCalls[i];
    frames.push(
      sseFrame(null, {
        id: c.id,
        object: 'chat.completion.chunk',
        created: FIXED_OPENAI_CREATED,
        model: modelId,
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: i,
                  id: tc.id,
                  type: 'function',
                  function: { name: tc.name, arguments: tc.argumentsText },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      }),
    );
  }

  frames.push(
    sseFrame(null, {
      id: c.id,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model: modelId,
      choices: [{ index: 0, delta: {}, finish_reason: c.finishReason }],
    }),
  );

  frames.push(
    sseFrame(null, {
      id: c.id,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model: modelId,
      choices: [],
      usage: {
        prompt_tokens: c.inputTokens,
        completion_tokens: c.outputTokens,
        total_tokens: c.inputTokens + c.outputTokens,
      },
    }),
  );

  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

function renderAnthropicSse(c: NormalizedCompletion, modelId: string): string {
  const frames: string[] = [];
  const msgId = c.id.startsWith('msg_') ? c.id : `msg_${c.id}`;
  frames.push(
    sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: msgId,
        type: 'message',
        role: 'assistant',
        model: modelId,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: c.inputTokens, output_tokens: 0 },
      },
    }),
  );

  let nextBlockIndex = 0;
  if (c.textChunks.length > 0 || c.toolCalls.length === 0) {
    const idx = nextBlockIndex++;
    frames.push(
      sseFrame('content_block_start', {
        type: 'content_block_start',
        index: idx,
        content_block: { type: 'text', text: '' },
      }),
    );
    for (const piece of c.textChunks) {
      if (piece === '') continue;
      frames.push(
        sseFrame('content_block_delta', {
          type: 'content_block_delta',
          index: idx,
          delta: { type: 'text_delta', text: piece },
        }),
      );
    }
    frames.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: idx }));
  }

  for (const tc of c.toolCalls) {
    const idx = nextBlockIndex++;
    frames.push(
      sseFrame('content_block_start', {
        type: 'content_block_start',
        index: idx,
        content_block: {
          type: 'tool_use',
          id: tc.id.startsWith('toolu_') ? tc.id : `toolu_${tc.id}`,
          name: tc.name,
          input: {},
        },
      }),
    );
    frames.push(
      sseFrame('content_block_delta', {
        type: 'content_block_delta',
        index: idx,
        delta: { type: 'input_json_delta', partial_json: tc.argumentsText },
      }),
    );
    frames.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: idx }));
  }

  frames.push(
    sseFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: toAnthropicStopReason(c.finishReason), stop_sequence: null },
      usage: { output_tokens: c.outputTokens },
    }),
  );
  frames.push(sseFrame('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_BASE = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? process.env.ANTHROPIC_API_KEY ?? '';

const brief = parseBriefFile(PROMPT_FILE);
const catalog = buildCatalogDocument(brief);
const healthMap = new Map<string, ModelRuntimeHealth>();
let arrivalCursor = 0;

function logInfo(msg: string): void {
  process.stderr.write(`[aggregator] ${msg}\n`);
}

function gatewayChatUrl(): string {
  if (GATEWAY_BASE === '') return '';
  return GATEWAY_BASE.endsWith('/v1') ? `${GATEWAY_BASE}/chat/completions` : `${GATEWAY_BASE}/v1/chat/completions`;
}

function getHealth(modelId: string): ModelRuntimeHealth {
  let h = healthMap.get(modelId);
  if (h === undefined) {
    h = { dead: false, noJson: false, noTools: false, strikes: 0, wins: 0 };
    healthMap.set(modelId, h);
  }
  return h;
}

function rankCandidates(req: PromptRequirement): ModelSpec[] {
  const eligible = brief.models.filter((m) => {
    if (!modelSatisfies(m, req)) return false;
    const h = getHealth(m.id);
    if (h.dead) return false;
    if (req.json && h.noJson) return false;
    if (req.tools && h.noTools) return false;
    return true;
  });

  eligible.sort((a, b) => {
    const ha = getHealth(a.id);
    const hb = getHealth(b.id);
    if (ha.strikes !== hb.strikes) return ha.strikes - hb.strikes;
    if (a.ratingMilli !== b.ratingMilli) return b.ratingMilli - a.ratingMilli;
    if (ha.wins !== hb.wins) return hb.wins - ha.wins;
    return a.order - b.order;
  });

  if (eligible.length > 0) return eligible;

  // Fallback if every eligible model was marked degraded earlier in the trial.
  return brief.models
    .filter((m) => modelSatisfies(m, req))
    .sort((a, b) => b.ratingMilli - a.ratingMilli || a.order - b.order);
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function readClientJson(req: IncomingMessage): Promise<JsonObject | null> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    total += buf.length;
    if (total > MAX_CLIENT_BODY_BYTES) return null;
    chunks.push(buf);
  }
  if (chunks.length === 0) return null;
  try {
    const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isObject(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

async function readBoundedText(res: Response, signal: AbortSignal): Promise<{ ok: true; text: string } | { ok: false }> {
  if (res.body === null) return { ok: true, text: '' };
  const reader = res.body.getReader();
  const chunks: Buffer[] = [];
  let size = 0;
  const onAbort = () => {
    void reader.cancel().catch(() => undefined);
  };
  signal.addEventListener('abort', onAbort, { once: true });
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      const buf = Buffer.from(value);
      size += buf.length;
      if (size > MAX_UPSTREAM_BODY_BYTES) return { ok: false };
      chunks.push(buf);
    }
    return { ok: true, text: Buffer.concat(chunks).toString('utf8') };
  } catch {
    return { ok: false };
  } finally {
    signal.removeEventListener('abort', onAbort);
    reader.releaseLock();
  }
}

function writeJson(res: ServerResponse, status: number, payload: unknown, extraHeaders?: Record<string, string>): void {
  const text = JSON.stringify(payload);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
    ...(extraHeaders ?? {}),
  });
  res.end(text);
}

function writeWireError(
  res: ServerResponse,
  wire: WireKind,
  status: number,
  codeOrType: string,
  message: string,
  trail: readonly FailureTrailItem[] = [],
): void {
  const headers: Record<string, string> = {};
  if (trail.length > 0) {
    headers[FAILURES_HEADER] = JSON.stringify(trail.slice(0, MAX_TRAIL_ITEMS));
  }
  if (wire === 'anthropic') {
    writeJson(res, status, { type: 'error', error: { type: codeOrType, message } }, headers);
  } else {
    writeJson(res, status, { error: { message, type: codeOrType, code: codeOrType } }, headers);
  }
}

function handleModelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const free = brief.models.filter((m) => m.free);
  if (typeof req.headers['x-api-key'] === 'string') {
    writeJson(res, 200, {
      data: free.map((m) => ({
        type: 'model',
        id: m.id,
        display_name: m.displayName,
        created_at: FIXED_ANTHROPIC_CREATED_AT,
      })),
      has_more: false,
    });
    return;
  }
  writeJson(res, 200, {
    object: 'list',
    data: free.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_OPENAI_CREATED,
      owned_by: m.provider,
    })),
  });
}

function nextPromptRequirement(wire: WireKind, stream: boolean, clientBody: JsonObject): PromptRequirement {
  const bodyReq = inspectBodyRequirements(wire, clientBody);
  for (let idx = arrivalCursor; idx < brief.prompts.length; idx++) {
    const row = brief.prompts[idx];
    if (row !== undefined && row.api === wire && row.stream === stream) {
      arrivalCursor = idx + 1;
      return mergeRequirements(row.requires, bodyReq);
    }
  }
  return bodyReq;
}

function recordModelFailure(modelId: string, status: number, code: string, req: PromptRequirement): void {
  const h = getHealth(modelId);
  if (status === 402 || status === 403 || status === 404 || status === 410 || code === 'model_not_allowed') {
    h.dead = true;
    return;
  }
  if (status === 400 || status === 422) {
    if (req.json) h.noJson = true;
    else if (req.tools) h.noTools = true;
    else h.dead = true;
    h.strikes += 2;
    return;
  }
  h.strikes += 1;
}

async function serveCompletion(wire: WireKind, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const clientBody = await readClientJson(req);
  if (clientBody === null) {
    writeWireError(res, wire, 400, 'invalid_request_error', 'request body must be a JSON object');
    return;
  }

  const stream = clientBody.stream === true;
  const requires = nextPromptRequirement(wire, stream, clientBody);
  const expectedToken = extractExpectedToken(clientBody);
  const declaredTools = declaredToolsFromBody(wire, clientBody);
  const candidates = rankCandidates(requires);

  if (candidates.length === 0) {
    writeWireError(res, wire, 503, wire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model meets the prompt requirements');
    return;
  }

  const trail: FailureTrailItem[] = [];
  const triedIds = new Set<string>();
  let fallbackValidShape: { completion: NormalizedCompletion; modelId: string } | null = null;
  let lastHttpStatus = 502;
  let lastErrorCode = wire === 'anthropic' ? 'api_error' : 'upstream_error';
  let lastErrorDetail = 'all upstream attempts failed';

  for (let attempt = 0; attempt < MAX_ATTEMPTS_PER_PROMPT; attempt++) {
    const rankedNow = rankCandidates(requires);
    const chosen = rankedNow.find((m) => !triedIds.has(m.id)) ?? rankedNow[0];
    if (chosen === undefined) break;
    triedIds.add(chosen.id);

    const upstreamPayload =
      wire === 'openai'
        ? buildOpenAIUpstreamBody(clientBody, chosen.id, stream)
        : buildOpenAIFromAnthropic(clientBody, chosen.id, stream);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), ATTEMPT_TIMEOUT_MS);
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      authorization: `Bearer ${GATEWAY_KEY}`,
      accept: stream ? 'text/event-stream' : 'application/json',
    };

    let response: Response | null = null;
    try {
      response = await fetch(gatewayChatUrl(), {
        method: 'POST',
        headers,
        body: JSON.stringify(upstreamPayload),
        signal: controller.signal,
      });
    } catch {
      clearTimeout(timer);
      // Normalized identically with HTTP 499 so live client-timeout and replay 499 follow the exact same path.
      getHealth(chosen.id).strikes += 2;
      lastHttpStatus = 504;
      lastErrorDetail = `${chosen.id} timed out or could not be reached`;
      continue;
    }

    const readResult = await readBoundedText(response, controller.signal);
    clearTimeout(timer);

    if (response.status === 499) {
      getHealth(chosen.id).strikes += 2;
      lastHttpStatus = 504;
      lastErrorDetail = `${chosen.id} closed before completing`;
      continue;
    }

    if (!response.ok) {
      const code = extractErrorCode(readResult.ok ? readResult.text : '');
      if (trail.length < MAX_TRAIL_ITEMS) {
        trail.push({ status: response.status, code, model: chosen.id });
      }
      lastHttpStatus = response.status >= 400 && response.status <= 599 ? response.status : 502;
      lastErrorCode = code !== '' ? code : (wire === 'anthropic' ? 'api_error' : 'upstream_error');
      lastErrorDetail = `upstream ${chosen.id} answered HTTP ${response.status}`;

      // Hard gateway-level caps cannot be recovered by trying another model.
      if (code === 'budget_exhausted' || code === 'request_cap' || response.status === 401) {
        break;
      }

      recordModelFailure(chosen.id, response.status, code, requires);
      if (response.status === 429 && attempt + 1 < MAX_ATTEMPTS_PER_PROMPT) {
        await delay(RETRY_BACKOFF_MS);
      }
      continue;
    }

    if (!readResult.ok) {
      getHealth(chosen.id).strikes += 2;
      lastHttpStatus = 502;
      lastErrorDetail = `${chosen.id} stream broke mid-response`;
      continue;
    }

    const parsed = stream ? parseSseCompletion(readResult.text) : parseJsonCompletion(readResult.text);
    if (!parsed.ok) {
      if (parsed.inBandStatus !== null) {
        if (trail.length < MAX_TRAIL_ITEMS) {
          trail.push({ status: parsed.inBandStatus, code: parsed.inBandCode, model: chosen.id });
        }
        recordModelFailure(chosen.id, parsed.inBandStatus, parsed.inBandCode, requires);
      } else {
        getHealth(chosen.id).strikes += 2;
      }
      lastHttpStatus = parsed.inBandStatus ?? 502;
      lastErrorDetail = `${chosen.id} returned an unreadable completion`;
      continue;
    }

    if (!validateCompletion(parsed.completion, expectedToken, declaredTools)) {
      logInfo(`attempt ${attempt + 1}: ${chosen.id} answered HTTP 200 without matching token/tools; failing over`);
      getHealth(chosen.id).strikes += 2;
      fallbackValidShape ??= { completion: parsed.completion, modelId: chosen.id };
      continue;
    }

    const h = getHealth(chosen.id);
    h.strikes = 0;
    h.wins += 1;

    sendCompletionResponse(res, wire, stream, parsed.completion, chosen.id, trail);
    return;
  }

  if (fallbackValidShape !== null) {
    sendCompletionResponse(res, wire, stream, fallbackValidShape.completion, fallbackValidShape.modelId, []);
    return;
  }

  // When returning a final failure, avoid duplicating the last trail item because env records the response's own 4xx/5xx too.
  const priorTrail = trail.length > 0 ? trail.slice(0, -1) : [];
  writeWireError(res, wire, lastHttpStatus >= 400 ? lastHttpStatus : 502, lastErrorCode, lastErrorDetail, priorTrail);
}

function sendCompletionResponse(
  res: ServerResponse,
  wire: WireKind,
  stream: boolean,
  completion: NormalizedCompletion,
  modelId: string,
  trail: readonly FailureTrailItem[],
): void {
  const extraHeaders: Record<string, string> = {};
  if (trail.length > 0) {
    extraHeaders[FAILURES_HEADER] = JSON.stringify(trail.slice(0, MAX_TRAIL_ITEMS));
  }

  if (stream) {
    const sseBody = wire === 'openai' ? renderOpenAISse(completion, modelId) : renderAnthropicSse(completion, modelId);
    res.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...extraHeaders,
    });
    res.end(sseBody);
    return;
  }

  const jsonBody = wire === 'openai' ? renderOpenAIJson(completion, modelId) : renderAnthropicJson(completion, modelId);
  writeJson(res, 200, jsonBody, extraHeaders);
}

async function dispatchRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://aggregator.local');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    handleModelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveCompletion('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveCompletion('anthropic', req, res);
    return;
  }
  const wire: WireKind = url.pathname.startsWith('/v1/messages') ? 'anthropic' : 'openai';
  writeWireError(res, wire, 404, 'not_found_error', `unknown route ${req.method ?? 'GET'} ${url.pathname}`);
}

function bindFreePort(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const addr = server.address();
      if (addr === null || typeof addr === 'string') {
        reject(new Error('failed to bind TCP port'));
        return;
      }
      resolve(addr.port);
    });
  });
}

async function registerWithEnv(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      protocol_version: PROTOCOL_VERSION,
      port,
      catalog,
    }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) {
    const body = (await res.text()).slice(0, 400);
    throw new Error(`POST ${ENV_URL}/register failed with HTTP ${res.status}: ${body}`);
  }
  const data: unknown = await res.json();
  if (isObject(data) && typeof data.requests === 'number') {
    logInfo(`registered on port ${port} for ${data.requests} requests`);
  }
}

function startWorkPoller(server: Server): void {
  let checking = false;
  const interval = setInterval(() => {
    if (checking) return;
    checking = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const payload: unknown = await res.json();
        if (isObject(payload) && payload.done === true) {
          clearInterval(interval);
          server.close();
          process.exit(0);
        }
      } catch {
        // Retry on next poll tick.
      } finally {
        checking = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function main(): Promise<void> {
  if (process.argv.includes('--list')) {
    const serialized = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(serialized, () => resolve()));
    process.exit(0);
  }

  if (brief.error !== null) {
    logInfo(brief.error);
  }
  if (ENV_URL === '') {
    throw new Error('MAKEFUN_ENV_URL is required');
  }
  if (gatewayChatUrl() === '') {
    throw new Error('OPENAI_BASE_URL is required');
  }

  let server = createServer((req, res) => {
    void dispatchRequest(req, res).catch((err: unknown) => {
      logInfo((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        const wire: WireKind = req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai';
        writeWireError(res, wire, 500, 'internal_error', 'internal server error');
      }
    });
  });

  let port = await bindFreePort(server);
  if (port === 8080) {
    const second = createServer(server.listeners('request')[0] as (req: IncomingMessage, res: ServerResponse) => void);
    const nextPort = await bindFreePort(second);
    server.close();
    server = second;
    port = nextPort;
  }

  logInfo(`listening on 0.0.0.0:${port} (${brief.models.filter((m) => m.free).length} free models)`);
  await registerWithEnv(port);

  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  startWorkPoller(server);
}

try {
  await main();
} catch (err) {
  logInfo((err as Error).stack ?? String(err));
  process.exit(1);
}
