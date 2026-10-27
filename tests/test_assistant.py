"""The AI assistant: what is sent to the model, and how the endpoint validates input.
A fake client stands in for the Anthropic API, so no network or API key is needed."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from model import MODEL_PATH
from webapp import assistant

RESULT = {
    'prediction': 'Tumor',
    'probabilities': {'No tumor': 0.1, 'Tumor': 0.9},
    'reliability': {'status': 'consistent', 'trace': []},
    'image': 'data:image/png;base64,AAAA',
    'heatmap': 'data:image/png;base64,BBBB',
}


class FakeClient:
    def __init__(self, stop_reason='end_turn', text='It looks consistent.'):
        self.calls = []
        reply = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type='text', text=text)])

        def create(**kwargs):
            self.calls.append(kwargs)
            return reply
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_sends_numbers_but_never_images():
    fake = FakeClient()
    answer = assistant.ask(RESULT, [{'role': 'user', 'content': 'Explain this.'}], client=fake)
    assert answer == 'It looks consistent.'
    call = fake.calls[0]
    opening = call['messages'][0]['content']
    assert '"Tumor": 0.9' in opening and opening.endswith('Explain this.')
    assert 'base64' not in opening
    assert 'project_documentation' in call['system']
    assert call['fallbacks'] == 'default'


def test_follow_up_turns_keep_the_result_in_the_first_message():
    fake = FakeClient()
    turns = [{'role': 'user', 'content': 'Explain.'}, {'role': 'assistant', 'content': 'Sure.'},
             {'role': 'user', 'content': 'Why?'}]
    assistant.ask(RESULT, turns, client=fake)
    sent = fake.calls[0]['messages']
    assert [m['role'] for m in sent] == ['user', 'assistant', 'user']
    assert '<analysis_result>' in sent[0]['content'] and sent[2]['content'] == 'Why?'


def test_refusal_returns_a_message_instead_of_empty_text():
    fake = FakeClient(stop_reason='refusal', text='')
    assert 'declined' in assistant.ask(RESULT, [{'role': 'user', 'content': 'Hi'}], client=fake)


@pytest.mark.skipif((Path.home() / '.config' / 'anthropic').exists(),
                    reason="a stored `ant auth login` profile would make a real API call")
def test_missing_credentials_become_a_setup_message(monkeypatch):
    for name in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_PROFILE'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(assistant, '_client', None)
    with pytest.raises(assistant.AssistantError) as error:
        assistant.ask(RESULT, [{'role': 'user', 'content': 'Hi'}])
    assert error.value.status == 503


def test_oversized_result_is_rejected():
    with pytest.raises(assistant.AssistantError) as error:
        assistant.result_context({'warnings': ['x' * 30_000]})
    assert error.value.status == 413


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="final weights missing")
def test_endpoint_rejects_a_conversation_ending_with_an_answer():
    from fastapi.testclient import TestClient
    from webapp.app import app
    with TestClient(app) as client:
        r = client.post('/api/ask', json={'result': RESULT, 'messages': [
            {'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello'}]})
        assert r.status_code == 422
        r = client.post('/api/ask', json={'result': RESULT, 'messages': []})
        assert r.status_code == 422
