import random


class Teeko2Player:
    """ An object representation for an AI game player for the game Teeko2.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a Teeko2Player object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this Teeko2Player object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        # detect drop phase
        drop_phase = self.is_drop_phase(state)

        if not drop_phase:          # move phase
            # use minimax with heuristic to get the state after a optimized move
            best_move = self.get_best_succ(state, d=3)
            move = []
            for i in range(5):
                for j in range(5):
                    # return the move list when both coordinates collected
                    if len(move) == 2:
                        return move
                    # compare the states before and after the move to form the move tuples
                    if best_move[i][j] == self.my_piece and state[i][j] != self.my_piece:
                        move.insert(0, (i, j))          # ensure the destination (row,col) tuple is at index 0
                    if best_move[i][j] != self.my_piece and state[i][j] == self.my_piece:
                        move.insert(1, (i, j))          # ensure the source (row,col) tuple is at index 1
        else:           # drop phase
            # if AI is the first one to drop, drop at the center
            starting = True
            for i in range(5):
                for j in range(5):
                    if state[i][j] != ' ':
                        starting = False;
            if starting:
                return [(2, 2)]

            # if not starting, use minimax with heuristic to get the state after a optimized move
            best_move = self.get_best_succ(state, d=3)
            for i in range(5):
                for j in range(5):
                    # compare the states before and after the move to form the move tuple
                    if best_move[i][j] == self.my_piece and state[i][j] != self.my_piece:
                        return [(i, j)]                 # only return the destination tuple during the drop phase

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this Teeko2Player object, or a generated successor state.

        Returns:
            int: 1 if this Teeko2Player wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and 3x3 square corners wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][col] == \
                        state[i + 3][col]:
                    return 1 if state[i][col] == self.my_piece else -1

        # TODO: check \ diagonal wins
        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col + 1] == state[row + 2][col + 2] == \
                        state[row + 3][col + 3]:
                    return 1 if state[row][col] == self.my_piece else -1

        # TODO: check / diagonal wins
        for row in range(3, 5):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row - 1][col + 1] == state[row - 2][col + 2] == \
                        state[row - 3][col + 3]:
                    return 1 if state[row][col] == self.my_piece else -1

        # TODO: check 3x3 square corners wins
        for row in range(1, 4):
            for col in range(1, 4):
                if state[row + 1][col + 1] != ' ' and state[row + 1][col + 1] == state[row - 1][col + 1] == \
                        state[row + 1][col - 1] == state[row - 1][col - 1]:
                    return 1 if state[row][col] == self.my_piece else -1

        return 0  # no winner yet

    def is_drop_phase(self, state):             # return true in drop phase, otherwise, false
        drop_phase = True
        cnt = 0
        for i in state:
            for j in i:
                if j != ' ':
                    cnt += 1
        if cnt == 8:
            drop_phase = False
        return drop_phase

    def on_board(self, r, c):           # return if the given coordinates is valid for a 5 * 5 board
        if (r in range(5)) and (c in range(5)):
            return True
        return False

    def deep_copy_board(self, state):           # return a deep copy of the given board
        rst = []
        for row in state:
            rst.append(row.copy())
        return rst

    def succ(self, state):              # find and return all successors of the given state
        drop_phase = self.is_drop_phase(state)          # detect drop phase

        succs = []
        if drop_phase:          # if in drop phase, dropping at any empty space is a valid successor
            for i in range(5):
                for j in range(5):
                    if state[i][j] == ' ':
                        temp = self.deep_copy_board(state)
                        temp[i][j] = self.my_piece
                        succs.append(temp)
        else:                   # if in moving phase, AI can only move one of its pieces to a nearby empty space
            for i in range(5):
                for j in range(5):
                    if state[i][j] == self.my_piece:
                        for r in range(i - 1, i + 2):
                            for c in range(j - 1, j + 2):
                                if self.on_board(r, c):             # make sure the destination is still on board
                                    if state[r][c] == ' ':
                                        temp = self.deep_copy_board(state)
                                        temp[r][c] = self.my_piece
                                        temp[i][j] = ' '
                                        succs.append(temp)
        return succs

    def heuristic_game_value(self, state):
        if self.game_value(state) != 0:
            return self.game_value(state)
        else:
            score = 0               # score of the current board, the more positive, the better for the AI
            # prefer center than edges
            for i in range(5):
                for j in range(5):
                    if (i in range(1, 4)) and (j in range(1, 4)):
                        if i == j == 2:                 # center
                            if state[i][j] == self.my_piece:
                                score += 15
                            elif state[i][j] == self.opp:
                                score -= 15
                        else:                           # inner circle
                            if state[i][j] == self.my_piece:
                                score += 10
                            elif state[i][j] == self.opp:
                                score -= 10

            # check each row
            for i in range(5):
                # looking for two consecutive pieces in the center 3 columns belonging to the same player
                for j in range(1, 4):
                    # if two consecutive my_piece found
                    if state[i][j] == state[i][j + 1] == self.my_piece:
                        # if an end is open, add points
                        if state[i][j - 1] != self.opp:
                            score += 2
                        if state[i][j + 1] != self.opp:
                            score += 2
                        # if both ends open, add more points
                        if state[i][j - 1] != self.opp and state[i][j + 1] != self.opp:
                            score += 5
                    # same for the opp's piece, except deducting points
                    elif state[i][j] == state[i][j + 1] == self.opp:
                        if state[i][j - 1] != self.my_piece:
                            score -= 2
                        if state[i][j + 1] != self.my_piece:
                            score -= 2
                        # penalize states more with two consecutive opp pieces with two open ends
                        if state[i][j - 1] != self.my_piece and state[i][j + 1] != self.my_piece:
                            score -= 10

                # looking for three consecutive pieces in a row on the left side belonging to the same player
                if state[i][0] != ' ' and state[i][0] == state[i][1] == state[i][2]:
                    if state[i][3] == ' ':          # open end
                        if state[i][0] == self.my_piece:            # my consecutive pieces are good
                            score += 10
                        else:                                       # opps' are bad
                            score -= 10
                    elif state[i][3] == self.my_piece:          # closed end with my piece, which is good
                        score += 15
                    else:                   # closed end with opp piece, which is bad
                        score -= 15
                # repeat for three consecutive pieces in a row on the right side
                if state[i][4] != ' ' and state[i][2] == state[i][3] == state[i][4]:
                    if state[i][1] == ' ':
                        if state[i][4] == self.my_piece:
                            score += 10
                        else:
                            score -= 10
                    elif state[i][1] == self.my_piece:
                        score += 15
                    else:
                        score -= 15
                # looking for three consecutive pieces in the center of a row belonging to the same player
                if state[i][1] == state[i][2] == state[i][3] == self.my_piece:      # if they are my pieces
                    if state[i][0] == ' ' and state[i][4] == ' ':       # very good if open ended on both side
                        score += 30
                    elif state[i][0] == ' ' or state[i][4] == ' ':      # still goof if one end is open
                        score += 15
                elif state[i][1] == state[i][2] == state[i][3] == self.opp:         # if they are opps', penalize
                    # penalize more to try to stop the opponent
                    if state[i][0] == ' ' and state[i][4] == ' ':
                        score -= 35
                    elif state[i][0] == ' ' or state[i][4] == ' ':
                        score -= 20

            # check each row, uncommented because this section is almost identical to the section above for the rows
            for j in range(5):
                for i in range(1, 4):
                    if state[i][j] == state[i + 1][j] == self.my_piece:
                        if state[i - 1][j] != self.opp:
                            score += 2
                        if state[i + 1][j] != self.opp:
                            score += 2
                        if state[i - 1][j] != self.opp and state[i + 1][j] != self.opp:
                            score += 5
                    elif state[i][j] == state[i + 1][j] == self.opp:
                        if state[i - 1][j] != self.my_piece:
                            score -= 2
                        if state[i + 1][j] != self.my_piece:
                            score -= 2
                        if state[i - 1][j] != self.my_piece and state[i + 1][j] != self.my_piece:
                            score -= 10
                if state[0][j] != ' ' and state[0][j] == state[1][j] == state[2][j]:
                    if state[3][j] == ' ':
                        if state[0][j] == self.my_piece:
                            score += 10
                        else:
                            score -= 10
                    elif state[3][j] == self.my_piece:
                        score += 15
                    else:
                        score -= 15
                if state[4][j] != ' ' and state[2][j] == state[3][j] == state[4][j]:
                    if state[1][j] == ' ':
                        if state[4][j] == self.my_piece:
                            score += 10
                        else:
                            score -= 10
                    elif state[1][j] == self.my_piece:
                        score += 15
                    else:
                        score -= 15
                if state[1][j] == state[2][j] == state[3][j] == self.my_piece:
                    if state[0][j] == ' ' and state[4][j] == ' ':
                        score += 30
                    elif state[0][j] == ' ' or state[4][j] == ' ':
                        score += 15
                elif state[1][j] == state[2][j] == state[3][j] == self.opp:
                    if state[0][j] == ' ' and state[4][j] == ' ':
                        score -= 35
                    elif state[0][j] == ' ' or state[4][j] == ' ':
                        score -= 20

            # check diagonals, looping through the center 9 squares
            for i in range(1, 4):
                for j in range(1, 4):
                    # if there is one piece of the AIs', go diagonally to find more consecutive pieces belongs to the AI
                    if state[i][j] == self.my_piece:
                        # award AI a little if there are two diagonally consecutive pieces belongs to the AI
                        if state[i - 1][j - 1] == self.my_piece:
                            score += 2
                        if state[i + 1][j - 1] == self.my_piece:
                            score += 2
                        if state[i - 1][j + 1] == self.my_piece:
                            score += 2
                        if state[i + 1][j + 1] == self.my_piece:
                            score += 2
                        # award AI a lot if there are three diagonally consecutive pieces belongs to the AI
                        if state[i - 1][j - 1] == self.my_piece and state[i + 1][j + 1] == self.my_piece:
                            score += 20
                        if state[i + 1][j - 1] == self.my_piece and state[i - 1][j + 1] == self.my_piece:
                            score += 20
                    # do the same for opps' pieces
                    elif state[i][j] == self.opp:
                        if state[i - 1][j - 1] == self.opp:
                            score -= 2
                        if state[i + 1][j - 1] == self.opp:
                            score -= 2
                        if state[i - 1][j + 1] == self.opp:
                            score -= 2
                        if state[i + 1][j + 1] == self.opp:
                            score -= 2
                        # penalize a lot more if there are three diagonally consecutive pieces belongs to the opp
                        if state[i - 1][j - 1] == self.opp and state[i + 1][j + 1] == self.opp:
                            score -= 25
                        if state[i + 1][j - 1] == self.opp and state[i - 1][j + 1] == self.opp:
                            score -= 25
            return score/1000

    def minimax(self, state, d, my_turn):
        mymax = -1
        mymin = 1
        val = self.game_value(state)
        if val != 0:                # if game ends at the given state, return the game value
            return val
        elif d == 0:                # if the search depth threshold is met, return a heuristic value of the given state
            return self.heuristic_game_value(state)
        else:                       # otherwise, continue recursion
            succs = self.succ(state)
            for succ in succs:
                # do minimax with depth d - 1 and flip turns on all successors of the current state
                t = self.minimax(succ, d - 1, not my_turn)
                # record max and min values of the successors
                if t > mymax:
                    mymax = t
                if t < mymin:
                    mymin = t
        if my_turn:
            return mymax        # return mymax at AI's turn
        else:
            return mymin        # return mymin at opp's turn

    def get_best_succ(self, state, d):          # return the best move of the AI calculated from a minimax approach
        succs = self.succ(state)
        scores = []
        for succ in succs:
            # evaluate each successor of the given state by the minimax algorithm with heuristics and a given depth
            scores.append(self.minimax(succ, d - 1, my_turn=False))
        # record all optimal successors
        mymax = max(scores)
        bests = []
        for i in range(len(scores)):
            if scores[i] == mymax:
                bests.append(succs[i])
        # break tie randomly if there are multiple best moves with the same highest score
        return bests[random.randint(0, len(bests) - 1)]


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = Teeko2Player()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved from " + chr(move[1][1] + ord("A")) + str(move[1][0]))
            print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                      (int(move_from[1]), ord(move_from[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
