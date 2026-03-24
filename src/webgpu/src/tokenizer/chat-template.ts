/**
 * Mini Jinja2 template interpreter for GGUF chat templates.
 * Supports the subset used by HuggingFace tokenizer.chat_template:
 *   - {% for item in list %}...{% endfor %}
 *   - {% if cond %}...{% elif cond %}...{% else %}...{% endif %}
 *   - {{ variable }} with dot access and filters (| trim)
 *   - String literals, ==, !=, not, and, or, is defined/undefined
 *   - loop.first, loop.last, loop.index0, loop.index, loop.length
 */

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface TemplateContext {
  messages: ChatMessage[];
  bos_token: string;
  eos_token: string;
  add_generation_prompt: boolean;
  [key: string]: unknown;
}

type Token =
  | { type: 'text'; value: string }
  | { type: 'expr'; value: string }
  | { type: 'tag'; value: string };

/** Tokenize a Jinja2 template string into text, expression, and tag tokens. */
function tokenize(template: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < template.length) {
    if (template.startsWith('{%', i)) {
      const end = template.indexOf('%}', i + 2);
      if (end < 0) break;
      tokens.push({ type: 'tag', value: template.substring(i + 2, end).trim() });
      i = end + 2;
    } else if (template.startsWith('{{', i)) {
      const end = template.indexOf('}}', i + 2);
      if (end < 0) break;
      tokens.push({ type: 'expr', value: template.substring(i + 2, end).trim() });
      i = end + 2;
    } else if (template.startsWith('{#', i)) {
      // Comment — skip to #}
      const end = template.indexOf('#}', i + 2);
      i = end < 0 ? template.length : end + 2;
    } else {
      let next = template.length;
      for (const marker of ['{%', '{{', '{#']) {
        const pos = template.indexOf(marker, i);
        if (pos >= 0 && pos < next) next = pos;
      }
      tokens.push({ type: 'text', value: template.substring(i, next) });
      i = next;
    }
  }
  return tokens;
}

/** Resolve a value from the context, supporting dot access (e.g. "message.role"). */
function resolve(expr: string, ctx: Record<string, unknown>): unknown {
  expr = expr.trim();

  // String literal
  if ((expr.startsWith("'") && expr.endsWith("'")) || (expr.startsWith('"') && expr.endsWith('"'))) {
    return expr.slice(1, -1);
  }
  // Boolean literals
  if (expr === 'true' || expr === 'True') return true;
  if (expr === 'false' || expr === 'False') return false;
  if (expr === 'none' || expr === 'None') return undefined;
  // Number
  if (/^\d+$/.test(expr)) return parseInt(expr, 10);

  // Dot access
  const parts = expr.split('.');
  let val: unknown = ctx;
  for (const part of parts) {
    if (val == null || typeof val !== 'object') return undefined;
    val = (val as Record<string, unknown>)[part];
  }
  return val;
}

/** Evaluate a simple expression, supporting filters (| trim), comparisons, not, and, or. */
function evaluate(expr: string, ctx: Record<string, unknown>): unknown {
  expr = expr.trim();

  // Handle "not" prefix
  if (expr.startsWith('not ')) {
    return !truthy(evaluate(expr.substring(4), ctx));
  }

  // Handle "and" / "or" (lowest precedence, split right-to-left)
  for (const op of [' or ', ' and ']) {
    const idx = expr.lastIndexOf(op);
    if (idx > 0) {
      const left = evaluate(expr.substring(0, idx), ctx);
      const right = evaluate(expr.substring(idx + op.length), ctx);
      return op === ' or ' ? (truthy(left) ? left : right) : (truthy(left) ? right : left);
    }
  }

  // Handle "is defined" / "is not defined"
  if (expr.endsWith(' is defined')) {
    const varName = expr.slice(0, -' is defined'.length).trim();
    return resolve(varName, ctx) !== undefined;
  }
  if (expr.endsWith(' is not defined')) {
    const varName = expr.slice(0, -' is not defined'.length).trim();
    return resolve(varName, ctx) === undefined;
  }

  // Comparison operators
  for (const op of ['!=', '==']) {
    const idx = expr.indexOf(op);
    if (idx > 0) {
      const left = evaluate(expr.substring(0, idx), ctx);
      const right = evaluate(expr.substring(idx + op.length), ctx);
      return op === '==' ? left === right : left !== right;
    }
  }

  // Filter: | trim
  if (expr.includes('|')) {
    const [base, ...filters] = expr.split('|');
    let val = evaluate(base, ctx);
    for (const f of filters) {
      const filter = f.trim();
      if (filter === 'trim' && typeof val === 'string') val = val.trim();
    }
    return val;
  }

  // Array index: messages[0]
  const bracketMatch = expr.match(/^(.+)\[(\d+)\]$/);
  if (bracketMatch) {
    const arr = resolve(bracketMatch[1], ctx);
    if (Array.isArray(arr)) return arr[parseInt(bracketMatch[2], 10)];
    return undefined;
  }

  // Length property
  if (expr.endsWith('.length') || expr.endsWith('|length')) {
    const base = expr.replace(/[.|]length$/, '');
    const val = resolve(base, ctx);
    if (Array.isArray(val)) return val.length;
    if (typeof val === 'string') return val.length;
    return 0;
  }

  return resolve(expr, ctx);
}

function truthy(val: unknown): boolean {
  if (val === undefined || val === null || val === false || val === 0 || val === '') return false;
  if (Array.isArray(val) && val.length === 0) return false;
  return true;
}

function toString(val: unknown): string {
  if (val === undefined || val === null) return '';
  return String(val);
}

/** Execute a parsed token stream and return the rendered string. */
function execute(tokens: Token[], ctx: Record<string, unknown>): string {
  let output = '';
  let i = 0;

  while (i < tokens.length) {
    const tok = tokens[i];

    if (tok.type === 'text') {
      output += tok.value;
      i++;
    } else if (tok.type === 'expr') {
      output += toString(evaluate(tok.value, ctx));
      i++;
    } else if (tok.type === 'tag') {
      const tag = tok.value;

      if (tag.startsWith('for ')) {
        // {% for VAR in ITERABLE %}...{% endfor %}
        const forMatch = tag.match(/^for\s+(\w+)\s+in\s+(.+)$/);
        if (!forMatch) { i++; continue; }
        const [, varName, iterExpr] = forMatch;
        const iterable = evaluate(iterExpr, ctx);
        const items = Array.isArray(iterable) ? iterable : [];

        // Collect body tokens until matching endfor
        const body: Token[] = [];
        let depth = 1;
        i++;
        while (i < tokens.length && depth > 0) {
          if (tokens[i].type === 'tag') {
            if (tokens[i].value.startsWith('for ')) depth++;
            else if (tokens[i].value === 'endfor') { depth--; if (depth === 0) { i++; break; } }
          }
          body.push(tokens[i]);
          i++;
        }

        // Execute body for each item
        for (let idx = 0; idx < items.length; idx++) {
          const loopCtx = {
            ...ctx,
            [varName]: items[idx],
            loop: {
              index0: idx,
              index: idx + 1,
              first: idx === 0,
              last: idx === items.length - 1,
              length: items.length,
            },
          };
          output += execute(body, loopCtx);
        }
      } else if (tag.startsWith('if ')) {
        // {% if COND %}...{% elif COND %}...{% else %}...{% endif %}
        const branches: { cond: string | null; body: Token[] }[] = [];
        let currentCond: string | null = tag.substring(3).trim();
        let currentBody: Token[] = [];
        let depth = 1;
        i++;

        while (i < tokens.length && depth > 0) {
          if (tokens[i].type === 'tag') {
            const inner = tokens[i].value;
            if (inner.startsWith('if ')) {
              depth++;
              currentBody.push(tokens[i]);
            } else if (inner === 'endif') {
              depth--;
              if (depth === 0) {
                branches.push({ cond: currentCond, body: currentBody });
                i++;
                break;
              }
              currentBody.push(tokens[i]);
            } else if (depth === 1 && inner.startsWith('elif ')) {
              branches.push({ cond: currentCond, body: currentBody });
              currentCond = inner.substring(5).trim();
              currentBody = [];
            } else if (depth === 1 && inner === 'else') {
              branches.push({ cond: currentCond, body: currentBody });
              currentCond = null; // else branch
              currentBody = [];
            } else {
              currentBody.push(tokens[i]);
            }
          } else {
            currentBody.push(tokens[i]);
          }
          i++;
        }

        // Evaluate branches
        for (const branch of branches) {
          if (branch.cond === null || truthy(evaluate(branch.cond, ctx))) {
            output += execute(branch.body, ctx);
            break;
          }
        }
      } else if (tag.startsWith('set ')) {
        // {% set VAR = EXPR %}
        const setMatch = tag.match(/^set\s+(\w+)\s*=\s*(.+)$/);
        if (setMatch) {
          ctx[setMatch[1]] = evaluate(setMatch[2], ctx);
        }
        i++;
      } else {
        // Unknown tag — skip
        i++;
      }
    } else {
      i++;
    }
  }

  return output;
}

/**
 * Apply a Jinja2-style chat template to a list of messages.
 * This is the main entry point — equivalent to HuggingFace's apply_chat_template().
 */
export function applyTemplate(
  template: string,
  messages: ChatMessage[],
  options?: {
    bos_token?: string;
    eos_token?: string;
    add_generation_prompt?: boolean;
  },
): string {
  const ctx: TemplateContext = {
    messages,
    bos_token: options?.bos_token ?? '',
    eos_token: options?.eos_token ?? '',
    add_generation_prompt: options?.add_generation_prompt ?? true,
  };

  const tokens = tokenize(template);
  return execute(tokens, ctx as unknown as Record<string, unknown>);
}
