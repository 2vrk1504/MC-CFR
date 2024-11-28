from games import PhantomTicTacToe, GameState
import numpy as np
from enum import Enum
from typing import List, Callable, Any, Dict
from tqdm import tqdm

class InfoSet:
	def __init__(self, game, player, actions, history_string):
		self.game = game
		self.player = player
		self.action_to_idx = {}
		self.idx_to_action = {}
		for i, action in enumerate(actions):
			self.action_to_idx[action] = i
			self.idx_to_action[i] = action
		self.next_infosets = [[] for _ in actions]
		self.history_string = history_string
		self.infoset_id = 0
		self.sequence_idx = 0
	
	def get_next_possible_infosets(self, action):
		return self.next_infosets[self.action_to_idx[int(action)]]
	
	def add_child(self, action, infoset):
		self.next_infosets[self.action_to_idx[int(action)]].append(infoset)
	
	def __repr__(self) -> str:
		return "P" + str(self.player+1) + "->" + self.history_string

class InfoSetVec:
	def __init__(self, game: "GameTFDP", player: int, *, default_value = 0.,
				 array: np.ndarray = None, dtype: np.dtype = np.float32):
		self.game = game
		self.player = player
		if array is not None:
			self.array = array.copy()
		else:
			size = game.infoset_nums[player]
			self.array = np.ones(size, dtype = dtype) * default_value

	def _array_operator_sanity_check(self, infoset: InfoSet) -> None:
		if self.game != infoset.game:
			raise ValueError("Game environment must match")
		if self.player != infoset.player:
			raise ValueError("Player must match")

	def __getitem__(self, infoset: InfoSet):
		self._array_operator_sanity_check(infoset)
		return self.array[infoset.infoset_id]

	def __setitem__(self, infoset: InfoSet, value: np.float32) -> None:
		self._array_operator_sanity_check(infoset)
		self.array[infoset.infoset_id] = value

	def __repr__(self) -> str:
		result = ""
		for infoset in self.game.infoset_list[self.player]:
			result += f"{infoset.history_string}, {self[infoset]} \n"
		return result

	def _math_opt_type_cast(self, other) -> "InfoSetVec":
		if isinstance(other, InfoSetVec):
			if self.game != other.game:
				raise ValueError("Game environment must match")
			if self.player != other.player:
				raise ValueError("Player must match")
			return other
		else:
			return InfoSetVec(self.game, self.player, default_value = other)

	def __neg__(self) -> "InfoSetVec":
		return InfoSetVec(self.game, self.player, array = -self.array)

	def __add__(self, other) -> "InfoSetVec":
		other = self._math_opt_type_cast(other)
		return InfoSetVec(self.game, self.player, array = self.array + other.array)

	def __radd__(self, other) -> "InfoSetVec":
		return self.__add__(other)

	def __sub__(self, other) -> "InfoSetVec":
		other = self._math_opt_type_cast(other)
		return InfoSetVec(self.game, self.player, array = self.array - other.array)

	def __rsub__(self, other) -> "InfoSetVec":
		other = self._math_opt_type_cast(other)
		return InfoSetVec(self.game, self.player, array = other.array - self.array)

	def __mul__(self, other) -> "InfoSetVec":
		other = self._math_opt_type_cast(other)
		return InfoSetVec(self.game, self.player, array = self.array * other.array)

	def __rmul__(self, other) -> "InfoSetVec":
		return self.__mul__(other)

	def __truediv__(self, other) -> "InfoSetVec":
		other = self._math_opt_type_cast(other)
		return InfoSetVec(self.game, self.player, array = self.array / other.array)

	def copy(self) -> "InfoSetVec":
		return InfoSetVec(self.game, self.player, array = self.array.copy())

	def seqvec(self) -> "SeqVec":
		vec = SeqVec(self.game, self.player, dtype = self.array.dtype)
		for infoset in self.game.infoset_list[self.player]:
			vec[infoset] = self[infoset]
		return vec

class SparseInfoSetVec:
	def __init__(self, game: "GameTFDP", player: int, non_zero: Dict[int, np.ndarray] = {}):
		self.game = game
		self.player = player
		self.non_zero = dict(non_zero)

	def _array_operator_sanity_check(self, infoset: InfoSet) -> None:
		if self.game != infoset.game:
			raise ValueError("Game environment must match")
		if self.player != infoset.player:
			raise ValueError("Player must match")

	def __getitem__(self, infoset: InfoSet):
		if isinstance(infoset, int):
			infoset = self.game.infoset_list[self.player][infoset]
		self._array_operator_sanity_check(infoset)
		if infoset.infoset_id in self.non_zero:
			return self.non_zero[infoset.infoset_id]
		else:
			return 0

	def __setitem__(self, infoset: InfoSet, value: np.float32) -> None:
		self._array_operator_sanity_check(infoset)
		self.non_zero[infoset.infoset_id] = value

	def __repr__(self) -> str:
		result = ""
		for infoset_id in self.non_zero:
			infoset = self.game.infoset_list[self.player][infoset_id]
			result += f"{infoset.history_string}, {self[infoset]} \n"
		return result

	def __neg__(self) -> "SparseInfoSetVec":
		res = {}
		for infoset_id in self.non_zero:
			res[infoset_id] = - self.non_zero[infoset_id]
		return SparseInfoSetVec(self.game, self.player, non_zero = res)

	def __add__(self, other) -> "SparseInfoSetVec":
		res = {}
		if isinstance(other, SparseInfoSetVec):
			for infoset_id in self.non_zero:
				res[infoset_id] = self[infoset_id] + other[infoset_id]
			for infoset_id in other.non_zero:				
				if infoset_id not in self.non_zero:
					res[infoset_id] = other[infoset_id]
		elif isinstance(other, InfoSetVec):
			my_seq_vec = self.infoset_vec()
			return my_seq_vec + other
		else:
			for infoset_id in self.non_zero:
				res[infoset_id] = self.non_zero[infoset_id] + other
		return SparseInfoSetVec(self.game, self.player, non_zero=res)

	def __radd__(self, other) -> "SparseInfoSetVec":
		return self.__add__(other)

	def __sub__(self, other) -> "SparseInfoSetVec":
		res = {}
		if isinstance(other, SparseInfoSetVec):
			for infoset_id in self.non_zero:
				res[infoset_id] = self[infoset_id] - other[infoset_id]
			for infoset_id in other.non_zero:
				if infoset_id not in self.non_zero:
					res[infoset_id] = -other[infoset_id]
		elif isinstance(other, InfoSetVec):
			my_seq_vec = self.infoset_vec()
			return my_seq_vec - other
		else:
			for infoset_id in self.non_zero:
				res[infoset_id] = self.non_zero[infoset_id] - other
		return SparseInfoSetVec(self.game, self.player, non_zero=res)

	def __rsub__(self, other) -> "SparseInfoSetVec":
		res = {}
		if isinstance(other, SparseInfoSetVec):
			for infoset_id in other.non_zero:
				res[infoset_id] = other[infoset_id] - self[infoset_id]
			for infoset_id in self.non_zero:
				if infoset_id not in other.non_zero:
					res[infoset_id] = -self[infoset_id]
		elif isinstance(other, InfoSetVec):
			my_seq_vec = self.infset_vec()
			return other - my_seq_vec
		else:
			for infoset_id in self.non_zero:
				res[infoset_id] = other - self[infoset_id]
		return SparseInfoSetVec(self.game, self.player, non_zero=res)

	def __mul__(self, other) -> "SparseInfoSetVec":
		res = {}
		if isinstance(other, SparseInfoSetVec) or isinstance(other, InfoSetVec):
			for infoset_id in self.non_zero:
				res[infoset_id] = self[infoset_id] * other[infoset_id]
		else:
			for infoset_id in self.non_zero:
				res[infoset_id] = self[infoset_id] * other
		return SparseInfoSetVec(self.game, self.player, non_zero=res)

	def __rmul__(self, other) -> "SparseInfoSetVec":
		return self.__mul__(other)

	def __truediv__(self, other) -> "SparseInfoSetVec":
		res = {}
		if isinstance(other, SparseInfoSetVec) or isinstance(other, InfoSetVec):
			for infoset_id in self.non_zero:
				res[infoset_id] = self[infoset_id] / other[infoset_id]
		else:
			for infoset_id in self.non_zero:
				res[infoset_id] = self[infoset_id] / other
		return SparseInfoSetVec(self.game, self.player, non_zero=res)

	def copy(self) -> "SparseInfoSetVec":
		res={}
		for infoset_id in self.non_zero:
			res[infoset_id] = self.non_zero[infoset_id].copy()
		return SparseInfoSetVec(self.game, self.player, non_zero=res)

	def infoset_vec(self) -> InfoSetVec:
		infosetvec = InfoSetVec(self.game, self.player)
		for infoset_id in self.non_zero:
			infoset = self.game.infoset_list[self.player][infoset_id]
			infosetvec[infoset] = self[infoset]
	
	def sparseseqvec(self) -> "SparseSeqVec":
		vec = SparseSeqVec(self.game, self.player)
		for infoset_id in self.non_zero:
			infoset = self.game.infoset_list[self.player][infoset_id]
			vec[infoset] = self[infoset] * np.ones(len(infoset.action_to_idx))
		return vec

class SeqVec:
	def __init__(self, game: "GameTFDP", player: int, *, default_value = 0.,
				 array: np.ndarray = None, dtype: np.dtype = np.float32):
		self.game = game
		self.player = player
		if array is not None:
			self.array = array.copy()

		else:
			size = game.num_sequences[player]
			self.array = np.ones(size, dtype = dtype) * default_value

	def _array_operator_sanity_check(self, infoset: InfoSet) -> None:
		if self.game != infoset.game:
			raise ValueError("Game environment must match")
		if self.player != infoset.player:
			raise ValueError("Player must match")

	def __getitem__(self, infoset: InfoSet) -> np.ndarray:
		self._array_operator_sanity_check(infoset)
		return self.array[infoset.sequence_idx: infoset.sequence_idx + len(infoset.idx_to_action)]

	def __setitem__(self, infoset: InfoSet, value: np.ndarray) -> None:
		self._array_operator_sanity_check(infoset)
		self.array[infoset.sequence_idx: infoset.sequence_idx + len(infoset.idx_to_action)] = value

	def __repr__(self) -> str:
		result = ""
		for infoset in self.game.infoset_list[self.player]:
			result += f"{infoset.history_string}, {self[infoset]} \n"
		return result

	def _math_opt_type_cast(self, other) -> "SeqVec":
		if isinstance(other, SeqVec):
			if self.game != other.game:
				raise ValueError("Game environment must match")
			if self.player != other.player:
				raise ValueError("Player must match")
			return other
		else:
			return SeqVec(self.game, self.player, default_value = other)

	def __neg__(self) -> "SeqVec":
		return SeqVec(self.game, self.player, array = -self.array)

	def __add__(self, other) -> "SeqVec":
		other = self._math_opt_type_cast(other)
		return SeqVec(self.game, self.player, array = self.array + other.array)

	def __radd__(self, other) -> "SeqVec":
		return self.__add__(other)

	def __sub__(self, other) -> "SeqVec":
		other = self._math_opt_type_cast(other)
		return SeqVec(self.game, self.player, array = self.array - other.array)

	def __rsub__(self, other) -> "SeqVec":
		other = self._math_opt_type_cast(other)
		return SeqVec(self.game, self.player, array = other.array - self.array)

	def __mul__(self, other) -> "SeqVec":
		other = self._math_opt_type_cast(other)
		return SeqVec(self.game, self.player, array = self.array * other.array)

	def __rmul__(self, other) -> "SeqVec":
		return self.__mul__(other)

	def __truediv__(self, other) -> "SeqVec":
		other = self._math_opt_type_cast(other)
		return SeqVec(self.game, self.player, array = self.array / other.array)

	def copy(self) -> "SeqVec":
		return SeqVec(self.game, self.player, array = self.array.copy())

	def reduce(self, func: Callable[[np.ndarray], Any]) -> InfoSetVec:
		result = InfoSetVec(self.game, self.player)
		for infoset in self.game.infoset_list[self.player]:
			result[infoset] = func(self[infoset])
		return result

	def map(self, func: Callable[[np.ndarray], np.ndarray]) -> "SeqVec":
		result = SeqVec(self.game, self.player)
		print(f"Mapping SeqVec for player {self.player}")
		for infoset in tqdm(self.game.infoset_list[self.player]):
			result[infoset] = func(self[infoset])
		return result

	def dot(self, other) -> float:
		other = self._math_opt_type_cast(other)
		return np.dot(self.array, other.array)
	
class SparseSeqVec:
	def __init__(self, game: "GameTFDP", player: int, non_zero: Dict[int, np.ndarray] = {}):
		self.game = game
		self.player = player
		self.non_zero = dict(non_zero)
	
	
	def _array_operator_sanity_check(self, infoset: InfoSet) -> None:
		if self.game != infoset.game:
			raise ValueError("Game environment must match")
		if self.player != infoset.player:
			raise ValueError("Player must match")

	def __getitem__(self, infoset: InfoSet) -> np.ndarray:
		if isinstance(infoset, int):
			infoset = self.game.infoset_list[self.player][infoset]
		self._array_operator_sanity_check(infoset)
		if infoset.infoset_id in self.non_zero:
			return self.non_zero[infoset.infoset_id]
		else:
			return np.zeros(len(infoset.action_to_idx))

	def __setitem__(self, infoset: InfoSet, value: np.ndarray) -> None:
		self._array_operator_sanity_check(infoset)
		if np.any(value != 0):
			self.non_zero[infoset.infoset_id] = value

	def __repr__(self) -> str:
		result = ""
		for infoset_id in self.non_zero:
			infoset = self.game.infoset_list[self.player][infoset_id]
			result += f"{infoset.history_string}, {self[infoset]} \n"
		return result

	def __neg__(self) -> "SparseSeqVec":
		res = {}
		for infoset_id in self.non_zero:
			res[infoset_id] = - self.non_zero[infoset_id]
		return SparseSeqVec(self.game, self.player, non_zero = res)

	def __add__(self, other) -> "SparseSeqVec":
		res = {}
		if isinstance(other, SparseSeqVec):
			for infoset_id in self.non_zero:
				res[infoset_id] = self.non_zero[infoset_id] + other[infoset_id]
			for infoset_id in other.non_zero:				
				if infoset_id not in self.non_zero:
					res[infoset_id] = other.non_zero[infoset_id]
		elif isinstance(other, SeqVec):
			my_seq_vec = self.seqvec()
			return my_seq_vec + other
		else:
			for infoset_id in self.non_zero:
				res[infoset_id] = self.non_zero[infoset_id] + other
		return SparseSeqVec(self.game, self.player, non_zero=res)

	def __radd__(self, other) -> "SparseSeqVec":
		return self.__add__(other)

	def __sub__(self, other) -> "SparseSeqVec":
		res = {}
		if isinstance(other, SparseSeqVec):
			for infoset_id in self.non_zero:
				res[infoset_id] = self.non_zero[infoset_id] - other[infoset_id]
			for infoset_id in other.non_zero:
				if infoset_id not in self.non_zero:
					res[infoset_id] = -other[infoset_id]
		elif isinstance(other, SeqVec):
			my_seq_vec = self.seqvec()
			return my_seq_vec - other
		else:
			for infoset_id in self.non_zero:
				res[infoset_id] = self.non_zero[infoset_id] - other
		return SparseSeqVec(self.game, self.player, non_zero=res)

	def __rsub__(self, other) -> "SparseSeqVec":
		res = {}
		if isinstance(other, SparseSeqVec):
			for infoset_id in other.non_zero:
				res[infoset_id] = other[infoset_id] - self[infoset_id]
			for infoset_id in self.non_zero:
				if infoset_id not in other.non_zero:
					res[infoset_id] = -self[infoset_id]
		elif isinstance(other, SeqVec):
			my_seq_vec = self.seqvec()
			return other - my_seq_vec
		else:
			for infoset_id in self.non_zero:
				res[infoset_id] = other - self.non_zero[infoset_id]
		return SparseSeqVec(self.game, self.player, non_zero=res)

	def __mul__(self, other) -> "SparseSeqVec":
		res = {}
		if isinstance(other, SparseSeqVec) or isinstance(other, SeqVec):
			for infoset_id in self.non_zero:
				res[infoset_id] = self.non_zero[infoset_id] * other[infoset_id]
		else:
			for infoset_id in self.non_zero:
				res[infoset_id] = self.non_zero[infoset_id] * other
		return SparseSeqVec(self.game, self.player, non_zero=res)

	def __rmul__(self, other) -> "SparseSeqVec":
		return self.__mul__(other)

	def __truediv__(self, other) -> "SparseSeqVec":
		res = {}
		if isinstance(other, SparseSeqVec) or isinstance(other, SeqVec):
			for infoset_id in self.non_zero:
				res[infoset_id] = self.non_zero[infoset_id] / other[infoset_id]
		else:
			for infoset_id in self.non_zero:
				res[infoset_id] = self.non_zero[infoset_id] / other
		return SparseSeqVec(self.game, self.player, non_zero=res)

	def copy(self) -> "SparseSeqVec":
		res={}
		for infoset_id in self.non_zero:
			res[infoset_id] = self.non_zero[infoset_id].copy()
		return SparseSeqVec(self.game, self.player, non_zero=res)

	def reduce(self, func: Callable[[np.ndarray], Any]) -> SparseInfoSetVec:
		result = SparseInfoSetVec(self.game, self.player)
		for infoset_id in self.non_zero:
			infoset = self.game.infoset_list[self.player][infoset_id]
			result[infoset] = func(self[infoset])
		return result

	def map(self, func: Callable[[np.ndarray], np.ndarray]) -> "SparseSeqVec":
		result = SparseSeqVec(self.game, self.player)
		for infoset_id in self.non_zero:
			result.non_zero[infoset_id] = func(self.non_zero[infoset_id])
		return result

	def dot(self, other) -> float:
		res = 0
		for infoset_id in self.non_zero:
			res += np.dot(self.non_zero[infoset_id], other[infoset_id])
		return res
	
	def seqvec(self) -> SeqVec:
		result = SeqVec(self.game, self.player)
		for infoset_id in self.non_zero:
			infoset = self.game.infoset_list[self.player][infoset_id]
			result[infoset] = self.non_zero[infoset_id]
		return result

class GameTFDP:
	def __init__(self, game_class = PhantomTicTacToe, num_players = 2):
		self.game_class = game_class
		self.num_players = num_players
		self.root = [None for _ in range(num_players)]
		self.infoset_map = [{} for _ in range(num_players)]
		self.infoset_list = [[] for _ in range(num_players)]
		self.num_infosets = [0 for _ in range(num_players)]
		self.num_sequences = [0 for _ in range(num_players)]
			
	def get_legal_actions(self, history_string):
		legal_actions = []
		for a in range(self.game_class.NUM_ACTIONS):
			if str(a) not in history_string:
				legal_actions.append(a)
		return legal_actions

	def build_tfdp_from_file(self, filename: str, player: int):

		with open(filename, 'r') as f:
			# instantiate all infosets
			print("Initializing Infosets...")
			for line in tqdm(f.readlines()):
				history_string = line.strip()
				actions = self.get_legal_actions(history_string)
				infoset = InfoSet(self, player, actions, history_string)
				infoset.infoset_id = self.num_infosets[player]
				infoset.sequence_idx = self.num_sequences[player]
				self.infoset_map[player][history_string] = infoset
				self.infoset_list[player].append(infoset)
				self.num_infosets[player] += 1
				self.num_sequences[player] += len(actions)
			
			# now connect them
			print("Making TFDP...")
			var = 2 if self.game_class == PhantomTicTacToe else 1
			for infoset in tqdm(self.infoset_list[player]):
				if len(infoset.history_string) > 1: # this is the root node check
					parent_infoset = self.infoset_map[player][infoset.history_string[:-var]]
					parent_action = infoset.history_string[-var]
					parent_infoset.add_child(parent_action, infoset)
					infoset.parent = parent_infoset
			self.root[player] = self.infoset_map[player]["|"]
			f.close()
		
	def __repr__(self):
		string = ""
		for player in range(self.num_players):
			string += f"\nPlayer {player}\n"
			seq = [self.root[player]]
			while seq:
				infoset = seq.pop(0)
				string += str(infoset)
				for sequences_to_visit in infoset.next_infosets:
					seq += sequences_to_visit
		return string
	
	def reachable_nodes(self):
		num_nodes = [0,0]
		for player in range(self.num_players):
			seq = [self.root[player]]
			while seq:
				num_nodes[player] += 1
				print(f"Num nodes visited for Pl{player}: {num_nodes[player]}", end = "\r")
				infoset = seq.pop(0)
				for sequences_to_visit in infoset.next_infosets:
					seq += sequences_to_visit
		print()
		return num_nodes
		
		

	