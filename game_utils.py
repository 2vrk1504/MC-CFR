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
									num_iters=1000, explore = "eps-uniform") -> Tuple[SeqVec, Dict[int, set]]:
	''' returns utility sparse vector and TFDP Subtree (contains only infosets that are visited)'''
	utilities = SparseSeqVec(game_tfdp, player)
	GameClass = game_tfdp.game_class
	
	# the subtree of visted states. The set contains the visited actions 
	tfdp_subtree = {game_tfdp.root[player].infoset_id: set()}
	
	# this is the outcome-sample based MC method
	# we will use uniform sampling strategy for player
	# we will average utility over 1000 runs of the game (not very expensive)
	for _ in range(num_iters):
		game_state = GameState.game_start()
		sample_importance = 1
		terminated = False
		while not terminated:
			try:
				curr_infoset_string = GameClass.state_information_set(game_state, game_state.curr_player)
				curr_infoset = game_tfdp.infoset_map[game_state.curr_player][curr_infoset_string]
				
				num_actions = len(curr_infoset.next_infosets)
				if player == game_state.curr_player:
					if curr_infoset.infoset_id not in tfdp_subtree:
						tfdp_subtree[curr_infoset.infoset_id] = set()
					# sample action uniformly -> have to change exploration policy
					if explore == "uniform":
						dist = normalize(np.ones(num_actions)) # uniform
					elif explore == "eps-uniform":
						eps = 0.8
						dist = eps * normalize(policies[game_state.curr_player][curr_infoset])\
							+ (1-eps)*normalize(np.ones(num_actions)) # epsilon uniform
					action_idx = np.random.choice(num_actions, p = dist)
					sample_importance *= dist[action_idx]
					tfdp_subtree[curr_infoset.infoset_id].add(action_idx)
				else:
					# sample action from other player policy
					dist = normalize(policies[game_state.curr_player][curr_infoset])
					action_idx = np.random.choice(num_actions, p=dist)
				
				game_state, terminated, util = GameClass.make_move(game_state, curr_infoset.idx_to_action[action_idx])
			except:
				print(f"Player: {game_state.curr_player}\t{game_state.history_string}\t{curr_infoset_string}")
				print(f"Infoset Actions: {curr_infoset.action_to_idx} \t Num Actions: {num_actions}")
				print(f"Next Infosets: {curr_infoset.next_infosets}")
				print(f"Dist: {policies[game_state.curr_player][curr_infoset]}")
				raise Exception 
		
		infoset_string, pl_action = GameClass.terminal_information_set(game_state, player)
		last_infoset = game_tfdp.infoset_map[player][infoset_string]
		action_idx = last_infoset.action_to_idx[pl_action]
		vec = np.zeros(len(last_infoset.action_to_idx)); vec[action_idx] = util[player]/sample_importance
		utilities[last_infoset] = utilities[last_infoset] + vec
	
	utilities /= num_iters
	return utilities, tfdp_subtree

def compute_counterfactual_value(game_tfdp: GameTFDP, player: int, policies: List[SparseSeqVec]) -> Tuple[SparseSeqVec, Dict[int, set]]:
	utilities, tfdp_subtree = compute_utility_vector(game_tfdp, player, policies)

	cf_value = SparseSeqVec(game_tfdp, player)
	def traverse(seq: List[InfoSet]):
		_ev = 0.
		for infoset in seq:
			# only visit the actions in subtree
			if infoset.infoset_id in tfdp_subtree:
				vec = np.zeros(len(infoset.action_to_idx))
				for i in tfdp_subtree[infoset.infoset_id]:
					vec[i] = traverse(infoset.next_infosets[i])
				cf_value[infoset] = vec + utilities[infoset]
				pl_policy = normalize(policies[player][infoset])
				_ev += cf_value[infoset].dot(pl_policy)
		return _ev

	traverse([game_tfdp.root[player]])
	return cf_value, tfdp_subtree

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

def get_reach_probability(game_tfdp: GameTFDP, player: int, policy: SparseSeqVec, tfdp_subtree: Dict[int, set] = {}) -> Union[SeqVec, SparseSeqVec]:
	flag = len(tfdp_subtree) > 0
	if flag:
		reach = SparseSeqVec(game_tfdp, player)
	else:
		reach = SeqVec(game_tfdp, player)

	def traverse(seq: List[InfoSet], agent_reach: float):
		try:
			for infoset in seq:
				# compute reaches only on subtree instead of entire tree (if we know how sparse utility is)
				action_set = []
				if flag:
					if infoset.infoset_id in tfdp_subtree:
						action_set = list(tfdp_subtree[infoset.infoset_id]) 
				else:
					action_set = list(infoset.idx_to_action)
				
				pl_policy = normalize(policy[infoset])
				vec = np.zeros(len(infoset.action_to_idx)); 
				vec[action_set] = pl_policy[action_set]
				reach[infoset] = agent_reach * vec
				for i in action_set:
					traverse(infoset.next_infosets[i], reach[infoset][i])
		except:
			print(f"Player: {player}, Infoset history: {infoset.history_string}")
			print(f"Infoset Actions: {infoset.action_to_idx} \t Num Actions: {len(infoset.action_to_idx)}")
			print(f"Next Infosets: {infoset.next_infosets}")
			print(f"Dist: {policy[infoset]}")
			raise Exception 
	
	traverse([game_tfdp.root[player]], agent_reach = 1.)
	return reach

def compute_policy_from_reach(game_tfdp: GameTFDP, player: int, reach: Union[SparseSeqVec, SeqVec]) -> Union[SeqVec, SparseSeqVec]:
	return reach.map(normalize)

def CFR(game_tfdp:GameTFDP, player: int, policies: List[SparseSeqVec],
									cf_regret: SparseSeqVec, cum_regret: SeqVec, tfdp_subtree: Dict[int, set]={}) -> SeqVec:
	flag = len(tfdp_subtree) > 0
	policies_next = policies[player].copy()
	def traverse(seq: List[InfoSet]):
		for infoset in seq:
			action_set = []
			if flag:
				if infoset.infoset_id in tfdp_subtree:
					action_set = list(tfdp_subtree[infoset.infoset_id]) 
			else:
				action_set = infoset.idx_to_action
			for i in action_set:
				traverse(infoset.next_infosets[i])
			cum_regret[infoset] = cum_regret[infoset] + cf_regret[infoset]
			policies_next[infoset] = normalize(cum_regret[infoset] + cf_regret[infoset])

	traverse([game_tfdp.root[player]])
	return policies_next

def compute_montecarlo_best_response(game_tfdp: GameTFDP, player: int, policies: List[SeqVec], utilities, tfdp_subtree) -> SeqVec:
	best_response = SparseSeqVec(game_tfdp, player)
	def traverse(seq: List[InfoSet]) -> float:
		max_ev = 0.
		for infoset in seq:
			local_max_ev = -np.inf
			best_i = None
			if infoset.infoset_id in tfdp_subtree:
				for i in tfdp_subtree[infoset.infoset_id]:
					max_ev_nxt = traverse(infoset.next_infosets[i]) + utilities[infoset][i]
					if max_ev_nxt > local_max_ev:
						local_max_ev = max_ev_nxt
						best_i = i
				max_ev += local_max_ev
			vec = np.zeros(len(infoset.action_to_idx)); vec[best_i] = 1.
			best_response[infoset] += vec
		return max_ev
	
	traverse([game_tfdp.root[player]])
		
	return best_response.map(normalize)

def compute_expected_value(game_tfdp: GameTFDP, player: int, policies: List[SeqVec]) -> float:
	# print("In Compute Expected Value")
	# print("Montecarlo utilities")
	utilities, tfdp_subtree = compute_utility_vector(game_tfdp, player, policies)
	# print("sparsity of utility")
	# print(f"non zero: {len(utilities.non_zero)}")
	# print("getting reach probabilities")
	ev = utilities.dot(get_reach_probability(game_tfdp, player, policies[player], tfdp_subtree))
	return ev

def compute_exploitability(game_tfdp: GameTFDP, player: int, policies: List[SeqVec]) -> float:
	utilities, tfdp_subtree = compute_utility_vector(game_tfdp, player, policies, num_iters=1000)
	best_response = compute_montecarlo_best_response(game_tfdp, player, policies, utilities, tfdp_subtree)
	# print(best_response)
	max_ev = utilities.dot(get_reach_probability(game_tfdp, player, best_response, tfdp_subtree))
	
	ev = utilities.dot(get_reach_probability(game_tfdp, player, policies[player], tfdp_subtree))
	# print(player, max_ev, ev)
	return max_ev - ev

def nash_conv(game_tfdp: GameTFDP, policies: List[SeqVec]):
	gaps = [compute_exploitability(game_tfdp, player, policies) for player in range(game_tfdp.num_players)]
	return np.sum(gaps), gaps

def prune_tfdp(game_tfdp: GameTFDP, policies: List[SparseSeqVec]):
	threshold = 1e-3
	initial_num_nodes = [0,0]
	pruned_num_nodes = [0,0]
	
	def traverse(seq: List[InfoSet], player_local: int):
		try:
			pruned_num_nodes[player_local] += 1
			for infoset in seq:
				dist = normalize(policies[player_local][infoset])
				idx_to_action_pruned = {}
				action_to_idx_pruned = {}
				next_infosets_pruned = []
				count = 0 # recounting the number of "good" actions
				for i in infoset.idx_to_action:
					if dist[i] > threshold:
						action = infoset.idx_to_action[i]
						idx_to_action_pruned[count] = action
						action_to_idx_pruned[action] = count
						count += 1
						next_infosets_pruned.append(infoset.next_infosets[i])
				# pruning done
				# the number of actions are now reduced to the value of "count"
				for next_seq in next_infosets_pruned:
					traverse(next_seq, player_local) 
				infoset.action_to_idx = action_to_idx_pruned
				infoset.idx_to_action = idx_to_action_pruned
				infoset.next_infosets = next_infosets_pruned
		except:
			print(f"player local: {player_local}, Player Global: {player}")
			print(f"Infoset: {infoset}, Infoset Actions: {infoset.action_to_idx}")
			print(f"policy: {dist}")
			raise Exception
	
	for player in range(game_tfdp.num_players):
		initial_num_nodes[player] = game_tfdp.num_infosets[player]
		traverse([game_tfdp.root[player]], player)
		
	print(f"Initial Num nodes: {initial_num_nodes}, After pruning: {pruned_num_nodes}")
	return