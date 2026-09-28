"""The localized intervention must not silently spill into subsequent tokens."""
from types import SimpleNamespace
import unittest
import torch
from sae_dashboard.causal_lab import prepare_case_inputs, generate_feature_report


class ContinuationTests(unittest.TestCase):
    def test_prefix_preserves_image_inputs_and_extends_masks(self):
        pixels = torch.randn(1, 3, 2, 2)
        original = {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.ones(1, 2, dtype=torch.long),
                    'token_type_ids': torch.tensor([[1, 0]]), 'pixel_values': pixels}
        app = SimpleNamespace(prepare_inputs=lambda *a: (original, original['input_ids'][0], 2),
                              processor=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **k: [3, 4])))
        inputs, ids, length = prepare_case_inputs(app, {'prompt': 'image task', 'image': 'example.png', 'assistant_prefix': 'Field:'})
        self.assertEqual(ids.tolist(), [1, 2, 3, 4])
        self.assertEqual(length, 2)  # Full displayed answer includes the shared prefix.
        self.assertIs(inputs['pixel_values'], pixels)
        self.assertEqual(inputs['attention_mask'].tolist(), [[1, 1, 1, 1]])
        self.assertEqual(inputs['token_type_ids'].tolist(), [[1, 0, 0, 0]])
        self.assertEqual(original['input_ids'].tolist(), [[1, 2]])

    def test_only_first_continuation_forward_changes_and_failure_removes_hook(self):
        layer = torch.nn.Identity()
        seen = []
        fail = False
        def generate(*args, **kwargs):
            for hidden in (torch.tensor([[[9., 9.], [1., 2.]]]), torch.tensor([[[3., 4.]]])):
                seen.append(layer(hidden))
            if fail:
                raise RuntimeError('failure')
            return 'prefix and continuation'
        app = SimpleNamespace(prepare_inputs=lambda *a: ({}, None, 2),
                              saes={17: SimpleNamespace(W_dec=torch.eye(2))},
                              get_layer_module=lambda _: layer, extract_hidden=lambda x: x,
                              replace_hidden=lambda old, new: new, generate_answer=generate)
        candidate = {'layer': 17, 'feature_id': 0, 'intervention_step': 2.}
        result = generate_feature_report(app, {'prompt': 'test'}, candidate, 1, schedule='First continuation step only')
        torch.testing.assert_close(seen[0], torch.tensor([[[9., 9.], [3., 2.]]]))
        torch.testing.assert_close(seen[1], torch.tensor([[[3., 4.]]]))
        self.assertEqual(result['intervention_count'], 1)
        self.assertEqual(result['forward_calls'], 2)
        seen.clear()
        result = generate_feature_report(app, {'prompt': 'test'}, candidate, 0, schedule='First continuation step only')
        self.assertEqual(result['intervention_count'], 0)
        torch.testing.assert_close(seen[0], torch.tensor([[[9., 9.], [1., 2.]]]))
        fail = True
        with self.assertRaises(RuntimeError):
            generate_feature_report(app, {'prompt': 'test'}, candidate, 1, schedule='First continuation step only')
        self.assertFalse(layer._forward_hooks)


class BoundaryTests(unittest.TestCase):
    def test_generated_prefix_keeps_original_ids_and_excludes_decision(self):
        from sae_dashboard.causal_lab import generate_baseline_prefix
        words = {1: 'Evidence', 2: '\nAction', 3: ':', 4: ' Review'}
        tok = SimpleNamespace(decode=lambda ids, **kw: ''.join(words[i] for i in ids))
        app = SimpleNamespace(prepare_inputs=lambda image, prompt: ({}, None, 2),
                              processor=SimpleNamespace(tokenizer=tok),
                              model=SimpleNamespace(generate=lambda **kw: torch.tensor([[90, 91, 1, 2, 3, 4]])))
        result = generate_baseline_prefix(app, 'image.png', 'prompt', 'Action:')
        self.assertEqual(result['assistant_prefix_ids'], [1, 2, 3])
        self.assertEqual(result['assistant_prefix'], 'Evidence\nAction:')
        self.assertEqual(result['baseline_full_answer'], 'Evidence\nAction: Review')
        with self.assertRaisesRegex(ValueError, 'did not contain'):
            generate_baseline_prefix(app, None, 'prompt', 'Absent:')
        with self.assertRaisesRegex(ValueError, 'inside a token'):
            generate_baseline_prefix(app, None, 'prompt', 'Evid')

    def test_common_profile_first_step_schedule_does_not_repeat(self):
        from unittest.mock import patch
        import demo_gradio_hackaton as app
        layer = torch.nn.Identity()
        seen = []
        def generate(**kwargs):
            seen.append(layer(torch.tensor([[[1., 2.]]])))
            seen.append(layer(torch.tensor([[[3., 4.]]])))
            return torch.tensor([[1, 2, 3]])
        with patch.object(app, 'LAYERS', [17]), patch.object(app, 'get_layer_module', return_value=layer), \
             patch.object(app, 'model', SimpleNamespace(generate=generate)), \
             patch.object(app, 'processor', SimpleNamespace(decode=lambda *a, **kw: 'answer')), \
             patch.object(app, 'extract_hidden', side_effect=lambda x: x), \
             patch.object(app, 'replace_hidden', side_effect=lambda old, new: new):
            app.generate_answer({}, 2, 16, 0, {17: torch.tensor([2., 0.])}, {17: 1}, first_step_only=True)
        torch.testing.assert_close(seen[0], torch.tensor([[[3., 2.]]]))
        torch.testing.assert_close(seen[1], torch.tensor([[[3., 4.]]]))
        self.assertFalse(layer._forward_hooks)
