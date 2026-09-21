"""The wire contract, control isolation and preservation of the frozen baseline."""
from copy import deepcopy
import hashlib
from pathlib import Path
import pytest
from test_candidates import raw, assessment
from jev_mario.candidates import (candidate_state, assessment_request, selection_request,
                                   realtime_request, assessments_from)
from jev_mario.rulebook import (PROFILES, CONTRACT_KEYS, GAME_RULES, REFERENCE,
                                attach_contract, contract_metadata)
from jev_mario import legacy_candidate_questions as legacy


def test_frozen_baseline_has_not_been_rewritten():
    path=Path(legacy.__file__)
    # Normalize Windows newlines; content was frozen before modifying the new path.
    assert hashlib.sha256(path.read_text(encoding='utf-8').encode('utf-8')).hexdigest() == '9fa87af556484ee63e28b61051964d1311b0eb1c47a65b4f332b9311b0f224ae'


@pytest.mark.parametrize('gap',[None,12,48])
def test_profiles_preserve_observation_candidates_and_schedule(gap):
    base=candidate_state(raw(gap))
    original=deepcopy(base)
    for profile in PROFILES:
        request=assessment_request(base,'test',profile)
        stripped={k:v for k,v in request['state'].items() if k not in CONTRACT_KEYS}
        assert stripped==base
        assert request['state']['candidates']==base['candidates']
        assert request['state']['decision_schedule']==base['decision_schedule']
        assert base==original


def test_legacy_requests_are_exactly_the_original_builders():
    base=candidate_state(raw(12))
    values=assessments_from(assessment(base),base)
    assert assessment_request(base,'test','legacy-v2')==legacy.assessment_request(base,'test')
    assert selection_request(base,values,'test','legacy-v2')==legacy.selection_request(base,values,'test')


def test_rules_only_adds_reference_and_rules_not_a_new_rubric():
    base=candidate_state(raw())
    old=assessment_request(base,'test','legacy-v2')
    new=assessment_request(base,'test','rules-only-v1')
    for key,q in old['questions'].items():
        assert new['questions'][key]=={**q,'instructions':REFERENCE+q['instructions']}
    values=assessments_from(assessment(base),base)
    old=selection_request(base,values,'test','legacy-v2')['questions']['action']
    new=selection_request(base,values,'test','rules-only-v1')['questions']['action']
    assert new=={**old,'instructions':REFERENCE+old['instructions']}


@pytest.mark.parametrize('profile',['rules-only-v1','rules-v1'])
def test_every_question_and_both_stages_receive_the_same_rules(profile):
    base=candidate_state(raw(12))
    values=assessments_from(assessment(base),base)
    first=assessment_request(base,'test',profile)
    second=selection_request(base,values,'test',profile)
    for key in CONTRACT_KEYS:
        assert first['state'].get(key)==second['state'].get(key)
    assert first['state']['game_rules']==GAME_RULES
    for q in [*first['questions'].values(),*second['questions'].values()]:
        assert REFERENCE in q['instructions']
    assert ('decision_policy' in first['state'])==(profile=='rules-v1')
    assert second['state']['jev_candidate_assessments']==values


def test_rule_copy_does_not_leak_edits_between_requests_or_profiles():
    base=candidate_state(raw())
    first=assessment_request(base,'test')
    first['state']['game_rules']['contact']['pit']='bogus'
    second=assessment_request(base,'test')
    assert second['state']['game_rules']['contact']['pit']==GAME_RULES['contact']['pit']
    assert 'game_rules' not in base
    assert not any(k in attach_contract(first['state'],'legacy-v2') for k in CONTRACT_KEYS)
    with pytest.raises(ValueError):
        attach_contract(base,'typo')


def test_metadata_changes_when_contract_changes_and_is_reproducible():
    first=contract_metadata('rules-v1')
    assert first==contract_metadata('rules-v1')
    assert first['contract_sha256']!=contract_metadata('rules-only-v1')['contract_sha256']


def test_realtime_is_one_choice_request_without_fabricated_assessments():
    base = candidate_state(raw(12))
    request = realtime_request(base, 'test', 'rules-v1')
    assert set(request['questions']) == {'action', 'jump_needed', 'danger'}
    assert request['state']['execution_mode'] == 'realtime_choice_one_call'
    assert 'jev_candidate_assessments' not in request['state']
    assert 'remains held while the next Choice request is in flight' in request['state']['control_contract']['buttons']
    assert 'stale by several frames' in request['state']['control_contract']['clock']
    text = request['questions']['action']['instructions']
    assert 'question block' in text and 'safe descending stomp' in text
    assert 'items_and_enemies' in request['state']['game_rules']
    assert set(request['questions']['action']['criteria']) == {
        'NOOP', 'RIGHT', 'RIGHT_A', 'RIGHT_B', 'RIGHT_A_B', 'A', 'LEFT'}


def test_preselected_fixture_hashes_and_all_profile_inputs_match():
    from jev_mario.prompt_evaluation import load_cases
    from jev_mario.rulebook import canonical_hash
    cases=load_cases(Path(__file__).parents[1]/'experiments/rule_prompt_cases.json')
    assert len(cases)==10
    for case in cases:
        for profile in PROFILES:
            wire=assessment_request(case['base_state'],'test',profile)['state']
            base={k:v for k,v in wire.items() if k not in CONTRACT_KEYS}
            assert canonical_hash(base)==case['base_state_sha256']
