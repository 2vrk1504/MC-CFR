class GameState:
	def __init__(self, player = 0, history_string = ""):
		self.history_string = history_string
		self.curr_player = player
	
	def game_start():
		return GameState()
	 	
class PhantomTicTacToe:
	EMPTY = "_"
	NUM_ACTIONS = 9

	def game_is_done(game_state: GameState):
		cell_states = PhantomTicTacToe.construct_cell_states(game_state.history_string)
		terminated = False
		winner = ''
		util = (0,0)
		# check if three in a row
		if cell_states[0] != PhantomTicTacToe.EMPTY and cell_states[0][0] == cell_states[3][0] and cell_states[3][0] == cell_states[6][0]:
			terminated = True
			winner = cell_states[0]
		
		elif cell_states[1] != PhantomTicTacToe.EMPTY and cell_states[1][0] == cell_states[4][0] and cell_states[4][0] == cell_states[7][0]:
			terminated = True
			winner = cell_states[1]

		elif cell_states[2] != PhantomTicTacToe.EMPTY and cell_states[2][0] == cell_states[5][0] and cell_states[5][0] == cell_states[8][0]:
			terminated = True
			winner = cell_states[2]

		elif cell_states[0] != PhantomTicTacToe.EMPTY and cell_states[0][0] == cell_states[1][0] and cell_states[1][0] == cell_states[2][0]:
			terminated = True
			winner = cell_states[0]

		elif cell_states[3] != PhantomTicTacToe.EMPTY and cell_states[3][0] == cell_states[4][0] and cell_states[4][0] == cell_states[5][0]:
			terminated = True
			winner = cell_states[3]

		elif cell_states[6] != PhantomTicTacToe.EMPTY and cell_states[6][0] == cell_states[7][0] and cell_states[7][0] == cell_states[8][0]:
			terminated = True
			winner = cell_states[6]

		elif cell_states[0] != PhantomTicTacToe.EMPTY and cell_states[0][0] == cell_states[4][0] and cell_states[4][0] == cell_states[8][0]:
			terminated = True
			winner = cell_states[0]

		elif cell_states[2] != PhantomTicTacToe.EMPTY and cell_states[2][0] == cell_states[4][0] and cell_states[4][0]== cell_states[6][0]:
			terminated = True
			winner = cell_states[2]
		
		# check if all cells occupied
		elif PhantomTicTacToe.EMPTY not in cell_states:
			terminated = True

		if winner != '':
			if winner == '0':
				util = (1, -1)
			else:
				util = (-1, 1)
		return terminated, util

	def construct_cell_states(history_string: str):
		cell_states = [PhantomTicTacToe.EMPTY for _ in range(PhantomTicTacToe.NUM_ACTIONS)]
		tokens = history_string.split('|')[:-1]
		for token in tokens:
			pl, action_observation = token.split(':')
			action, observation = action_observation
			if cell_states[int(action)] == PhantomTicTacToe.EMPTY:
				cell_states[int(action)] = pl
		return cell_states
		
	def make_move(game_state: GameState, action: int):
		history_string = game_state.history_string
		cell_states = PhantomTicTacToe.construct_cell_states(history_string)
		if cell_states[action] == PhantomTicTacToe.EMPTY:
			cell_states[action] = str(game_state.curr_player)
			history_string += f"{game_state.curr_player}:{action}*|"
						
		elif cell_states[action] != PhantomTicTacToe.EMPTY:
			assert str(game_state.curr_player) not in cell_states[action]
			cell_states[action] += str(game_state.curr_player)
			history_string += f"{game_state.curr_player}:{action}.|"
		
		next_game_state = GameState(1 - game_state.curr_player, history_string)			
		terminated, util = PhantomTicTacToe.game_is_done(next_game_state)
		return next_game_state, terminated, util
		
	def terminal_information_set(game_state: GameState, player: int):
		# infoset and action that led to this state for Player
		tokens = game_state.history_string.split('|')[:-1]
		infoset_string = "|"
		for token in tokens:
			pl, action_observation = token.split(':')
			if player == int(pl):
				infoset_string += action_observation

		# remove the last action observation from infoset
		player_action = int(infoset_string[-2])
		infoset_string = infoset_string[:-2]

		return infoset_string, player_action
	
	def state_information_set(game_state: GameState, player:int):
		# information state the current game state is in for Player
		tokens = game_state.history_string.split('|')[:-1]
		infoset_string = "|"
		for token in tokens:
			pl, action_observation = token.split(':')
			if player == int(pl):
				infoset_string += action_observation

		return infoset_string
	
class RockPaperSuperScissors:
	NUM_ACTIONS = 3
	TURNS = 2
	def game_is_done(game_state: GameState):
		tokens = game_state.history_string.split('|')[:-1]
		terminated = False; util = (0,0)
		if len(tokens) == 2*RockPaperSuperScissors.TURNS:
			terminated = True; util = [0,0]
			for turn in range(RockPaperSuperScissors.TURNS):
				_, fpa = tokens[0 + 2*turn].split(":"); fpa = int(fpa)
				_, spa = tokens[1 + 2*turn].split(":"); spa = int(spa)
				if fpa == spa:
					util[0] += 0; util[1] += 0
				elif fpa == 0:
					if spa == 1:
						util[0] += -1; util[1] += 1
					elif spa == 2:
						util[0] += 1; util[1] += -1
				elif fpa == 1:
					if spa == 0:
						util[0] += 1; util[1] += -1
					elif spa == 2:
						util[0] += -2; util[1] += 2
				elif fpa == 2:
					if spa == 0:
						util[0] += -1; util[1] += 1
					elif spa == 1:
						util[0] += 2; util[1] += -2
		
		return terminated, tuple(util)
	
	def make_move(game_state: GameState, action: int):
		history_string = game_state.history_string
		assert len(history_string.split("|")[:-1]) < 2*RockPaperSuperScissors.TURNS, "Exceeded max number of moves"
		history_string += f"{game_state.curr_player}:{action}|"
		next_game_state = GameState(1-game_state.curr_player, history_string)
		terminated, util = RockPaperSuperScissors.game_is_done(next_game_state)
		return next_game_state, terminated, util

	def terminal_information_set(game_state: GameState, player: int):
		# infoset and action that led to this state for Player
		tokens = game_state.history_string.split('|')[:-1]
		infoset_string = "|"
		for token in tokens:
			pl, action_observation = token.split(':')
			if player == int(pl):
				infoset_string += action_observation

		# remove the last action observation from infoset
		player_action = int(infoset_string[-1])
		infoset_string = infoset_string[:-1]

		return infoset_string, player_action
		
	def state_information_set(game_state: GameState, player:int):
		# information state the current game state is in for Player
		tokens = game_state.history_string.split('|')[:-1]
		infoset_string = "|"
		for token in tokens:
			pl, action_observation = token.split(':')
			if player == int(pl):
				infoset_string += action_observation

		return infoset_string