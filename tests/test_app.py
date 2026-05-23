import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


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


from unittest.mock import patch, MagicMock


def test_ask_returns_sources_field():
    with patch('app.find_relevant_chunks', return_value=[
        (0.94, '[Page 4, Source: test.pdf] Some text here'),
        (0.81, '[Page 12, Source: other.pdf] More text'),
    ]), patch('app.generate_text', return_value='Answer text'), \
       patch('app.get_cached_response', return_value=None), \
       patch('app.get_response_from_db', return_value=None), \
       patch('app.store_query_response'), patch('app.cache_response'):
        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={'question': 'test question'})
        data = res.get_json()
        assert res.status_code == 200
        assert 'response' in data
        assert 'sources' in data
        assert isinstance(data['sources'], list)


def test_ask_cached_response_has_empty_sources():
    with patch('app.get_cached_response', return_value='Cached answer'):
        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.post('/ask', data={'question': 'test question'})
        data = res.get_json()
        assert data['response'] == 'Cached answer'
        assert data['sources'] == []


def test_documents_returns_grouped_list():
    mock_records = [
        MagicMock(payload={'source': 'alpha.pdf'}),
        MagicMock(payload={'source': 'alpha.pdf'}),
        MagicMock(payload={'source': 'beta.pdf'}),
    ]
    with patch('app.qdrant') as mock_qdrant:
        mock_qdrant.scroll.return_value = (mock_records, None)
        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.get('/documents')
        data = res.get_json()
        assert res.status_code == 200
        assert 'documents' in data
        by_name = {d['filename']: d for d in data['documents']}
        assert by_name['alpha.pdf']['chunks'] == 2
        assert by_name['beta.pdf']['chunks'] == 1


def test_documents_empty_collection():
    with patch('app.qdrant') as mock_qdrant:
        mock_qdrant.scroll.return_value = ([], None)
        import app as flask_app
        flask_app.app.config['TESTING'] = True
        client = flask_app.app.test_client()
        res = client.get('/documents')
        data = res.get_json()
        assert data['documents'] == []
