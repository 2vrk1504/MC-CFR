from games import PhantomTicTacToe, GameState
import numpy as np
from typing import List, Tuple, Dict, Union
from game_tree import  (InfoSet, InfoSetVec, SeqVec, SparseInfoSetVec, SparseSeqVec, GameTFDP)
import pickle


def normalize(vec: np.array):
	vec_positive = vec * (vec > 0)
	vec_sum = np.sum(vec_positive)
	if vec_sum > 1e-9:
		return vec_positive / vec_sum
	else:
		return np.ones_like(vec) / len(vec)

def compute_utility_vector(game_tfdp: GameTFDP, player: int, policies: List[SparseSeqVec],
									num_iters=10) -> SparseSeqVec:
	''' returns utility sparse vector and TFDP Subtree (contains only infosets that are visited)'''
	utilities = SparseSeqVec(game_tfdp, player)
	GameClass = game_tfdp.game_class
	# this is the External-sample based MC method
	# sample actions of opp
	# we will average utility over 1000 runs of the game (not very expensive)
	for _ in range(num_iters):
		def traverse(game_state: GameState, ext_reach: float) -> float:
			_ev = 0.
			terminated, util = GameClass.game_is_done(game_state)
			if not terminated:
				curr_infoset_string = GameClass.state_information_set(game_state, game_state.curr_player)
				curr_infoset = game_tfdp.infoset_map[game_state.curr_player][curr_infoset_string]
				if game_state.curr_player == player:
					vec = np.zeros(len(curr_infoset.idx_to_action))
					for i in curr_infoset.idx_to_action:
						next_game_state, _, _ = GameClass.make_move(game_state, curr_infoset.idx_to_action[i])
						vec[i] = traverse(next_game_state, ext_reach)
					utilities[curr_infoset] += vec
				else:
					# sample action of opponent
					dist = normalize(policies[game_state.curr_player][curr_infoset])
					action_idx = np.random.choice(len(curr_infoset.idx_to_action), p=dist)
					next_game_state, _, _ = GameClass.make_move(game_state, curr_infoset.idx_to_action[action_idx])
					_ev += traverse(next_game_state, ext_reach * dist[action_idx])
			else:
				_ev = ext_reach * util[player]
			return _ev

		traverse(GameState.game_start(), ext_reach = 1.)
	
	utilities /= num_iters
	return utilities

def compute_counterfactual_value(game_tfdp: GameTFDP, player: int, policies: List[SparseSeqVec]) -> SparseSeqVec:
	utilities = compute_utility_vector(game_tfdp, player, policies)

	cf_value = SparseSeqVec(game_tfdp, player)
	def traverse(seq: List[InfoSet]):
		_ev = 0.
		for infoset in seq:
			vec = np.zeros(len(infoset.idx_to_action))
			for i in infoset.idx_to_action:
				vec[i] = traverse(infoset.next_infosets[i])
			cf_value[infoset] = vec + utilities[infoset]
			pl_policy = normalize(policies[player][infoset])
			_ev += cf_value[infoset].dot(pl_policy)
		return _ev

	traverse([game_tfdp.root[player]])
	return cf_value

def get_uniform_policy(game_tfdp: GameTFDP, player: int) -> SeqVec:
	uniform_policy = None
	try:
		with open(f"player{player}_uniform_policy.pkl", "rb") as f:
			uniform_policy = pickle.load(f)
	except:
		uniform_policy = SeqVec(game_tfdp, player).map(normalize)
		with open(f"player{player}_uniform_policy.pkl", "wb") as f:
			pickle.dump(uniform_policy, f, pickle.HIGHEST_PROTOCOL)
	return uniform_policy

def get_reach_probability(game_tfdp: GameTFDP, player: int, policy: SparseSeqVec) -> Union[SeqVec, SparseSeqVec]:

	reach = SparseSeqVec(game_tfdp, player)

	def traverse(seq: List[InfoSet], agent_reach: float):
		for infoset in seq:
			# compute reaches only on subtree instead of entire tree (if we know how sparse utility is)
			pl_policy = normalize(policy[infoset])
			reach[infoset] = agent_reach * pl_policy
			for i in infoset.idx_to_action:
				traverse(infoset.next_infosets[i], reach[infoset][i])

	traverse([game_tfdp.root[player]], agent_reach = 1.)
	return reach

def compute_policy_from_reach(game_tfdp: GameTFDP, player: int, reach: Union[SparseSeqVec, SeqVec]) -> Union[SeqVec, SparseSeqVec]:
	return reach.map(normalize)

def CFR(game_tfdp:GameTFDP, player: int, policies: List[SparseSeqVec],
									cf_regret: SparseSeqVec, cum_regret: SeqVec) -> SeqVec:
	policies_next = policies[player].copy()
	def traverse(seq: List[InfoSet]):
		for infoset in seq:
			for i in infoset.idx_to_action:
				traverse(infoset.next_infosets[i])
			cum_regret[infoset] = cum_regret[infoset] + cf_regret[infoset]
			policies_next[infoset] = normalize(cum_regret[infoset] + cf_regret[infoset])

	traverse([game_tfdp.root[player]])
	return policies_next

def compute_montecarlo_best_response(game_tfdp: GameTFDP, player: int, policies: List[SeqVec], utilities) -> SeqVec:
	# utilities = compute_utility_vector(game_tfdp, player, policies, num_iters=1000)
	best_response = SeqVec(game_tfdp, player)

	def traverse(seq: List[InfoSet]) -> float:
		max_ev = 0.
		for infoset in seq:
			local_max_ev = -np.inf
			best_i = None
			for i in infoset.idx_to_action:
				max_ev_nxt = traverse(infoset.next_infosets[i]) + utilities[infoset][i]
				if max_ev_nxt > local_max_ev:
					local_max_ev = max_ev_nxt
					best_i = i
			max_ev += local_max_ev
			best_response[infoset][best_i] = 1.
		return max_ev

	traverse([game_tfdp.root[player]])
	return best_response

def compute_expected_value(game_tfdp: GameTFDP, player: int, policies: List[SeqVec]) -> float:
	# print("In Compute Expected Value")
	# print("Montecarlo utilities")
	utilities = compute_utility_vector(game_tfdp, player, policies)
	# print("sparsity of utility")
	# print(f"non zero: {len(utilities.non_zero)}")
	# print("getting reach probabilities")
	ev = utilities.dot(get_reach_probability(game_tfdp, player, policies[player]))
	return ev

def compute_exploitability( game_tfdp: GameTFDP, player: int, policies: List[SeqVec]) -> float:
	utilities = compute_utility_vector(game_tfdp, player, policies, num_iters=10)
	best_response = compute_montecarlo_best_response( game_tfdp, player, policies, utilities)
	# print(best_response)
	max_ev = utilities.dot(get_reach_probability(game_tfdp, player, best_response))
	# print(max_ev)
	ev = utilities.dot(get_reach_probability(game_tfdp, player, policies[player]))
	# print(ev)
	return max_ev - ev

def nash_conv(game_tfdp: GameTFDP, policies: List[SeqVec]):
	gaps = [compute_exploitability(game_tfdp, player, policies) for player in range(game_tfdp.num_players)]
	return np.sum(gaps), gaps

