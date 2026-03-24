import { describe, it, expect } from 'vitest';
import { applyTemplate, type ChatMessage } from '../src/tokenizer/chat-template.js';

const userMsg: ChatMessage[] = [{ role: 'user', content: 'Hello' }];
const multiTurn: ChatMessage[] = [
  { role: 'user', content: 'Hi' },
  { role: 'assistant', content: 'Hello!' },
  { role: 'user', content: 'How are you?' },
];

describe('Chat Template Engine', () => {
  describe('ChatML template', () => {
    const template = `{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}<|im_start|>assistant
`;

    it('renders single message', () => {
      const result = applyTemplate(template, userMsg);
      expect(result).toContain('<|im_start|>user');
      expect(result).toContain('Hello');
      expect(result).toContain('<|im_end|>');
      expect(result).toContain('<|im_start|>assistant');
    });

    it('renders multi-turn', () => {
      const result = applyTemplate(template, multiTurn);
      expect(result).toContain('<|im_start|>user');
      expect(result).toContain('<|im_start|>assistant');
      expect(result).toContain('Hello!');
      expect(result).toContain('How are you?');
    });
  });

  describe('Simple Llama 2 template', () => {
    const template = `{% for message in messages %}{% if message.role == 'user' %}[INST] {{ message.content }} [/INST]{% else %}{{ message.content }}{% endif %}{% endfor %}`;

    it('wraps user messages in [INST]', () => {
      const result = applyTemplate(template, userMsg);
      expect(result).toContain('[INST]');
      expect(result).toContain('Hello');
      expect(result).toContain('[/INST]');
    });

    it('does not wrap assistant messages', () => {
      const result = applyTemplate(template, multiTurn);
      expect(result).toContain('[INST] Hi [/INST]');
      expect(result).toContain('Hello!');
      expect(result).toContain('[INST] How are you? [/INST]');
    });
  });

  describe('Template with bos/eos tokens', () => {
    const template = `{{ bos_token }}{% for message in messages %}{{ message.role }}: {{ message.content }}{{ eos_token }}{% endfor %}`;

    it('inserts bos and eos tokens', () => {
      const result = applyTemplate(template, userMsg, { bos_token: '<s>', eos_token: '</s>' });
      expect(result).toBe('<s>user: Hello</s>');
    });
  });

  describe('Template with conditionals', () => {
    const template = `{% for message in messages %}{% if message.role == 'system' %}SYSTEM: {{ message.content }}{% elif message.role == 'user' %}USER: {{ message.content }}{% else %}BOT: {{ message.content }}{% endif %}
{% endfor %}`;

    it('handles if/elif/else', () => {
      const msgs: ChatMessage[] = [
        { role: 'system', content: 'Be helpful' },
        { role: 'user', content: 'Hi' },
        { role: 'assistant', content: 'Hello' },
      ];
      const result = applyTemplate(template, msgs);
      expect(result).toContain('SYSTEM: Be helpful');
      expect(result).toContain('USER: Hi');
      expect(result).toContain('BOT: Hello');
    });
  });

  describe('Template with loop variables', () => {
    const template = `{% for message in messages %}{% if loop.first %}[FIRST]{% endif %}{{ message.content }}{% if loop.last %}[LAST]{% endif %}{% endfor %}`;

    it('detects first and last', () => {
      const result = applyTemplate(template, multiTurn);
      expect(result).toContain('[FIRST]Hi');
      expect(result).toContain('How are you?[LAST]');
    });
  });

  describe('Template with add_generation_prompt', () => {
    const template = `{% for message in messages %}{{ message.content }}{% endfor %}{% if add_generation_prompt %}GENERATE{% endif %}`;

    it('includes generation prompt by default', () => {
      const result = applyTemplate(template, userMsg);
      expect(result).toContain('GENERATE');
    });

    it('excludes when disabled', () => {
      const result = applyTemplate(template, userMsg, { add_generation_prompt: false });
      expect(result).not.toContain('GENERATE');
    });
  });

  describe('Template with trim filter', () => {
    const template = `{{ "  hello  " | trim }}`;

    it('trims whitespace', () => {
      const result = applyTemplate(template, []);
      expect(result).toBe('hello');
    });
  });

  describe('Template with set', () => {
    const template = `{% set name = 'world' %}Hello {{ name }}`;

    it('sets and uses variables', () => {
      const result = applyTemplate(template, []);
      expect(result).toBe('Hello world');
    });
  });

  describe('Template with not operator', () => {
    const template = `{% if not add_generation_prompt %}NO{% else %}YES{% endif %}`;

    it('handles not', () => {
      const result = applyTemplate(template, [], { add_generation_prompt: true });
      expect(result).toBe('YES');
    });
  });
});
