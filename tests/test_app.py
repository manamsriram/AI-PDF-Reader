import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from functools import wraps
from unittest.mock import patch, MagicMock


# ---- Helpers ----

def _make_auth_decorator(user_id='test-user-id'):
    """require_auth stand-in that injects g.user_id without Supabase."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            from flask import g
            g.user_id = user_id
            return f(*args, **kwargs)
        return wrapper
    return decorator


def _supabase_chain(data=None):
    """MagicMock satisfying chained Supabase builder calls ending in .execute()."""
    mock = MagicMock()
    mock.table.return_value = mock
    mock.select.return_value = mock
    mock.eq.return_value = mock
    mock.order.return_value = mock
    mock.limit.return_value = mock
    mock.insert.return_value = mock
    mock.delete.return_value = mock
    mock.execute.return_value = MagicMock(data=data or [])
    return mock


# ---- Sigmoid tests ----

def test_sigmoid_midpoint():
    from app import _sigmoid
    assert abs(_sigmoid(0.0) - 0.5) < 1e-6


def test_sigmoid_positive():
    from app import _sigmoid
    assert _sigmoid(2.0) > 0.5
    assert _sigmoid(2.0) < 1.0


def test_sigmoid_negative():
    from app import _sigmoid
    assert _sigmoid(-2.0) < 0.5
    assert _sigmoid(-2.0) > 0.0


# ---- /ask tests ----

def test_ask_returns_response_and_sources():
    """Single-turn ask returns response text and parsed sources."""
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain()), \
         patch('app.find_relevant_chunks', return_value=[
             (0.94, '[Page 4, Source: test.pdf] Some text here'),
             (0.81, '[Page 12, Source: other.pdf] More text'),
         ]), \
         patch('app.generate_text', return_value='Answer text'), \
         patch('app.get_cached_response', return_value=(None, None)), \
         patch('app.cache_response'):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={'question': 'test question', 'session_id': ''})
        data = res.get_json()

        assert res.status_code == 200
        assert data['response'] == 'Answer text'
        assert isinstance(data['sources'], list)
        assert len(data['sources']) == 2
        assert data['sources'][0]['source'] == 'test.pdf'
        assert data['sources'][0]['page'] == 4


def test_ask_cached_response_returned_directly():
    """Cache hit returns cached answer without calling LLM."""
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain()), \
         patch('app.get_cached_response', return_value=('Cached answer', [])), \
         patch('app.generate_text') as mock_gen:

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={'question': 'test question'})
        data = res.get_json()

        assert res.status_code == 200
        assert data['response'] == 'Cached answer'
        assert data['sources'] == []
        mock_gen.assert_not_called()


def test_ask_missing_question_returns_400():
    """Blank question returns 400."""
    with patch('app.require_auth', _make_auth_decorator()):
        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={'question': '   '})
        assert res.status_code == 400
        assert 'error' in res.get_json()


def test_ask_no_auth_returns_401():
    """No auth header → 401."""
    import app as flask_app
    flask_app.app.config['TESTING'] = True
    client = flask_app.app.test_client()
    res = client.post('/ask', data={'question': 'hello'})
    assert res.status_code == 401


def test_ask_multi_turn_passes_history_to_llm():
    """Prior session turns are fetched and forwarded to generate_text."""
    prior_turns = [{'question': 'What is X?', 'answer': 'X is a thing.'}]
    captured = {}

    def fake_generate_text(prompt, conversation_history=None):
        captured['history'] = conversation_history
        return 'Follow-up answer'

    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain(data=prior_turns)), \
         patch('app.find_relevant_chunks', return_value=[
             (0.9, '[Page 1, Source: doc.pdf] Some context'),
         ]), \
         patch('app.generate_text', side_effect=fake_generate_text), \
         patch('app.get_cached_response', return_value=(None, None)), \
         patch('app.cache_response'):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={
            'question': 'Can you elaborate?',
            'session_id': 'session-abc-123',
        })

        assert res.status_code == 200
        assert captured.get('history') == prior_turns


def test_ask_saves_session_id_to_history():
    """New response is inserted to query_history with the session_id."""
    inserted = {}

    supabase_mock = _supabase_chain(data=[])

    original_table = supabase_mock.table.side_effect

    def capturing_table(name):
        m = MagicMock()
        m.select.return_value = m
        m.eq.return_value = m
        m.order.return_value = m
        m.limit.return_value = m
        m.execute.return_value = MagicMock(data=[])
        def capturing_insert(record):
            if name == 'query_history':
                inserted.update(record)
            r = MagicMock()
            r.execute.return_value = MagicMock()
            return r
        m.insert = capturing_insert
        return m

    supabase_mock.table = capturing_table

    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', supabase_mock), \
         patch('app.find_relevant_chunks', return_value=[
             (0.9, '[Page 1, Source: doc.pdf] Context'),
         ]), \
         patch('app.generate_text', return_value='Answer'), \
         patch('app.get_cached_response', return_value=(None, None)), \
         patch('app.cache_response'):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={
            'question': 'What is Y?',
            'session_id': 'session-xyz',
        })

        assert res.status_code == 200
        assert inserted.get('session_id') == 'session-xyz'
        assert inserted.get('question') == 'What is Y?'


# ---- /history tests ----

def test_history_returns_sessions_grouped():
    """Rows grouped by session_id; most recent session first."""
    rows = [
        {'id': 'r1', 'question': 'Q1', 'answer': 'A1', 'sources': [], 'created_at': '2026-01-01T10:00:00', 'session_id': 'sess-1'},
        {'id': 'r2', 'question': 'Q2', 'answer': 'A2', 'sources': [], 'created_at': '2026-01-01T10:01:00', 'session_id': 'sess-1'},
        {'id': 'r3', 'question': 'Q3', 'answer': 'A3', 'sources': [], 'created_at': '2026-01-01T11:00:00', 'session_id': 'sess-2'},
    ]
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain(data=rows)):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.get('/history')
        data = res.get_json()

        assert res.status_code == 200
        assert 'sessions' in data
        sessions = data['sessions']
        assert len(sessions) == 2
        # Most recent session first
        assert sessions[0]['session_id'] == 'sess-2'
        assert sessions[1]['session_id'] == 'sess-1'
        assert len(sessions[1]['questions']) == 2


def test_history_legacy_rows_become_singleton_sessions():
    """Rows without session_id each form their own session keyed by row id."""
    rows = [
        {'id': 'leg-1', 'question': 'Q1', 'answer': 'A1', 'sources': [], 'created_at': '2026-01-01T09:00:00', 'session_id': None},
        {'id': 'leg-2', 'question': 'Q2', 'answer': 'A2', 'sources': [], 'created_at': '2026-01-01T09:01:00', 'session_id': None},
    ]
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain(data=rows)):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.get('/history')
        data = res.get_json()

        assert res.status_code == 200
        assert len(data['sessions']) == 2
        for session in data['sessions']:
            assert len(session['questions']) == 1


def test_history_no_auth_returns_401():
    """No auth header → 401."""
    import app as flask_app
    flask_app.app.config['TESTING'] = True
    client = flask_app.app.test_client()
    res = client.get('/history')
    assert res.status_code == 401


# ---- /documents tests ----

def test_documents_returns_list():
    """Documents endpoint returns filename and chunk count from Supabase."""
    rows = [
        {'filename': 'alpha.pdf', 'chunk_count': 5},
        {'filename': 'beta.pdf', 'chunk_count': 2},
    ]
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain(data=rows)):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.get('/documents')
        data = res.get_json()

        assert res.status_code == 200
        by_name = {d['filename']: d for d in data['documents']}
        assert by_name['alpha.pdf']['chunks'] == 5
        assert by_name['beta.pdf']['chunks'] == 2


def test_documents_empty_returns_empty_list():
    """No documents → empty list, not an error."""
    with patch('app.require_auth', _make_auth_decorator()), \
         patch('app.supabase_admin', _supabase_chain(data=[])):

        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.get('/documents')
        data = res.get_json()

        assert res.status_code == 200
        assert data['documents'] == []


def test_documents_no_auth_returns_401():
    """No auth header → 401."""
    import app as flask_app
    flask_app.app.config['TESTING'] = True
    client = flask_app.app.test_client()
    res = client.get('/documents')
    assert res.status_code == 401
