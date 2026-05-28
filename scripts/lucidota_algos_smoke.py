#!/usr/bin/env python3
"""Smoke tests for local algorithm primitives."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ALGOS import (
    bandit_router,
    bayes_update,
    cockpit_metrics,
    counterfactual_effects,
    decreasing_pruning,
    distributed_leader_election,
    fisher_localization,
    fold_change_detection,
    gini_coefficient,
    capybara_optimization,
    chelydrid_ambush,
    hdc,
    label_foundry,
    honeybee_store,
    infotaxis,
    minhash,
    perceptual_dedupe,
    poikilotherm_schoolfield,
    privacy,
    possum_filter,
    minimum_cost_tree,
    nlms,
    pheromone,
    physarum_network,
    rbf_surrogate,
    regret_engine,
    rsa_cipher,
    semantic_neighbors,
    serpentina_self_righting,
    shannon_entropy,
    sketches,
    sparse_wta,
    ssim,
    thanatosis,
    temporal_motifs,
    tri_algo_conduit,
    doomsday_calendar,
    endpoint_circuit_breaker,
    hoeffding_tree,
    voronoi_partition,
)


def main()->int:
    a=hdc.symbol_vector('a', 512)
    b=hdc.symbol_vector('b', 512)
    pair=hdc.bind(a,b)
    sim_self=hdc.similarity(pair,pair)
    weights, err=nlms.update([0.0,0.0],[1.0,2.0],1.0)
    visual=ssim.ssim([0,1,2,3],[0,1,2,3])
    s1=minhash.signature(minhash.shingles('alpha beta gamma delta epsilon'))
    s2=minhash.signature(minhash.shingles('alpha beta gamma delta epsilon'))
    mh=minhash.similarity(s1,s2)
    probs=pheromone.probabilities([pheromone.EdgeSignal('a', 0.1), pheromone.EdgeSignal('b', 1.0)])
    leaders=distributed_leader_election.maximal_independent_set({'a': {'b'}, 'b': {'a','c'}, 'c': {'b'}}, seed=1)
    fcd=fold_change_detection.response_series([1.0, 2.0, 4.0])[-1]
    region=voronoi_partition.nearest((0.1,0.1), [(0,0),(10,10)])
    tree=minimum_cost_tree.tree_cost({'r':(0,0),'a':(1,0),'b':(1,1)}, [('r','a'),('a','b')], 'r')
    q=physarum_network.flux(1.0, 2.0, 3.0, 1.0)
    store, delta=honeybee_store.update_store(1.0, [2.0], [0.5])
    tag=sparse_wta.top_k_mask(sparse_wta.expand([1,2,3], 32), 4)
    action=infotaxis.best_action({'a': (0.5, [0.9,0.1], [0.5,0.5]), 'b': (0.5, [0.5,0.5], [0.5,0.5])})
    angle=fisher_localization.best_angle([-1.0, 0.0, 1.0], 0.0, 1.0)
    p0=decreasing_pruning.prune_probability(0.0)
    p5=decreasing_pruning.prune_probability(5.0)
    filtered=possum_filter.filter_entities([
        possum_filter.Entity('a', 49.0, -123.0, 'coffee', 0.9),
        possum_filter.Entity('b', 49.0001, -123.0001, 'coffee', 0.8),
        possum_filter.Entity('c', 49.0001, -123.0001, 'bakery', 0.7),
    ], delta_m=50)
    model=rbf_surrogate.fit([(0.0,), (1.0,), (2.0,)], [1.0, 0.0, 1.0], epsilon=1.0)
    best_x,best_y=rbf_surrogate.grid_search(model, [(0.0,), (1.0,), (2.0,)])
    anneal=thanatosis.decide(0.5, k=20, t0=1.0, alpha=0.9, seed=1)
    social=capybara_optimization.social_interaction([1.0, 2.0], [3.0, 4.0], k=1, r1=0.5)
    evaded=capybara_optimization.predator_evasion([10.0], delta=0.2, r2=1.0)
    eps=hoeffding_tree.hoeffding_bound(1.0, 1e-7, 1000)
    split=hoeffding_tree.should_split(0.80, 0.30, 1.0, 1e-7, 1000)
    cold=poikilotherm_schoolfield.normalized_activity(5.0)
    warm=poikilotherm_schoolfield.normalized_activity(25.0)
    turtle=serpentina_self_righting.Morphology(length=30.0, width=24.0, height=12.0, mass=8.0)
    si=serpentina_self_righting.sphericity_index(turtle.length, turtle.width, turtle.height)
    fi=serpentina_self_righting.flatness_index(turtle.length, turtle.width, turtle.height)
    righting=serpentina_self_righting.righting_time_index(turtle, neck_lever=1.5)
    strike=chelydrid_ambush.integrate_strike(chelydrid_ambush.pulse_force(10.0, 5), dt=0.01, m_head=1.0, drag_cd=0.1, fluid_density=1.0, area=1.0)
    marginal=bayes_update.bayes_marginal(0.01,0.95,0.05)
    posterior=bayes_update.bayes_update(0.01,0.95,marginal)
    entropy=shannon_entropy.shannon_entropy(['a','b','c','d'])
    gini=gini_coefficient.gini_coefficient([0,0,0,10])
    rsa_round=rsa_cipher.rsa_decrypt(rsa_cipher.rsa_encrypt(42,17,3233),2753,3233)
    weekday=doomsday_calendar.doomsday(2026,5,3)
    honesty=cockpit_metrics.cockpit_honesty(9,1)
    bandit_router.reset_policy(); bandit_router.update_policy([bandit_router.BanditUpdate('c','go',1.0,0.5)])
    bandit=bandit_router.select_action({'pressure':2.0}, ['go','wait'])
    strat=regret_engine.compute_regret_weighted_strategy([regret_engine.MathAction('a',2.0), regret_engine.MathAction('b',1.0)], [])
    anon=privacy.anonymize_for_indexing({'email':'x@y','note':'ok'})
    cms=sketches.count_min_sketch(['a','a','b'], width=8, depth=2)
    ph=perceptual_dedupe.hamming_distance(perceptual_dedupe.compute_phash([1,2,3,4]), perceptual_dedupe.compute_phash([1,2,3,4]))
    breaker=endpoint_circuit_breaker.EndpointCircuitBreaker(2); breaker.record_failure(); breaker.record_failure()
    lf=[label_foundry.LabelingFunctionResult('lf','d',1)]
    pl=label_foundry.aggregate_labels([lf])[0]
    semantic_neighbors.clear_enclave(); semantic_neighbors.register_document('a',[1,0]); semantic_neighbors.register_document('b',[0.9,0.1])
    neighbor=semantic_neighbors.semantic_neighbors('a')[0][0]
    sessions=temporal_motifs.sessionize_events([{'t':0,'type':'x'},{'t':1,'type':'x'},{'t':4000,'type':'y'}])
    burst=temporal_motifs.detect_bursts([{'type':'x'},{'type':'x'},{'type':'y'}])[0]
    ce=counterfactual_effects.estimate_causal_effect('T','Y',[], {'T':[0,0,1,1], 'Y':[1,1,3,3]})
    conduit=tri_algo_conduit.decide((b"<html><a href=\"x\">signal metadata high value</a></html>" * 50), observations=512, status_code=200, mime="text/html", structural_links=50)
    ok=(
        sim_self == 1.0 and len(weights)==2 and err == 1.0 and visual > 0.999 and mh == 1.0
        and probs[1][1] > probs[0][1] and leaders and fcd[0] > 0 and region == 0
        and tree > 0 and q == 1.0 and store > 1.0 and delta > 0 and sum(tag) == 4
        and action == 'a' and abs(angle) == 1.0 and p0 > p5
        and [e.id for e in filtered] == ['a','c'] and best_x == (1.0,) and best_y < 1e-6
        and 0.0 <= anneal.probability <= 1.0 and social == [2.0, 3.0] and evaded == [12.0]
        and eps > 0 and split.should_split
        and 0 <= cold < warm <= 1.0 and 0 < si < 1 and fi > 1 and righting > 0
        and strike.peak_velocity > 0 and strike.distance > 0
        and abs(marginal-0.059) < 1e-9 and posterior > 0.16 and entropy == 2.0
        and abs(gini-0.75) < 1e-9 and rsa_round == 42 and weekday == 0 and honesty == 0.9
        and bandit.action_id in {'go','wait'} and strat['a'] > strat['b'] and anon['email'] == '<redacted>'
        and cms[0] and ph == 0 and not breaker.allow() and pl.label == 1 and neighbor == 'b'
        and len(sessions) == 2 and burst.key == 'x' and ce.ate_estimate == 2.0 and conduit.action == 'burst'
    )
    print(json.dumps({
        'ok':ok,
        'algorithm_modules':40,
        'hdc_self_similarity':sim_self,
        'nlms_error':err,
        'ssim_identity':visual,
        'minhash_identity':mh,
        'pheromone_b_probability':probs[1][1],
        'leader_count':len(leaders),
        'infotaxis_action':action,
        'possum_kept':[e.id for e in filtered],
        'opossum_best':best_x,
        'thanatosis_probability':anneal.probability,
        'capybara_social':social,
        'hoeffding_epsilon':eps,
        'poikilotherm_warm_activity':warm,
        'serpentina_flatness':fi,
        'chelydrid_strike_peak':strike.peak_velocity,
        'math_zip_modules':16,
        'bayes_posterior':posterior,
        'entropy_uniform':entropy,
        'bandit_action':bandit.action_id,
        'causal_ate':ce.ate_estimate,
        'tri_algo_action':conduit.action,
    }, sort_keys=True))
    return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
