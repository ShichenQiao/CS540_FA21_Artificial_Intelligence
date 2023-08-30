import random
import time
from importlib import reload
import game


debug = True


class TestTeeko2Player:

    def __init__(self):
        super().__init__()

    def test_gameplay(self):
        reload(game)
        ai = game.Teeko2Player()

        piece_count = 0
        turn = 0

        # drop phase
        while piece_count < 8 and ai.game_value(ai.board) == 0:
            # get the AI's move
            if ai.my_piece == ai.pieces[turn]:
                # ai.print_board()

                move = ai.make_move(ai.board)

                ai.place_piece(move, ai.my_piece)
                # print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
            else:
                # Choose a random legal move
                # # select an unoccupied space randomly
                move = []
                (row, col) = (random.randint(0,4), random.randint(0,4))
                while not ai.board[row][col] == ' ':
                    (row, col) = (random.randint(0,4), random.randint(0,4))

                ai.opponent_move([(row, col)])

            # update the game variables
            piece_count += 1
            turn += 1
            turn %= 2
            if debug:
                ai.print_board()

        # move phase - can't have a winner until all 8 pieces are on the board
        while ai.game_value(ai.board) == 0:

            # get the player or AI's move
            if ai.my_piece == ai.pieces[turn]:
                # ai.print_board()
                move = ai.make_move(ai.board)

                ai.place_piece(move, ai.my_piece)
                # print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
                # print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
            else:
                possible_moves = []
                for r in range(5):
                    for c in range(5):
                        if ai.board[r][c] == ai.opp:
                            for i in [-1, 0, 1]:
                                for j in [-1, 0, 1]:
                                    if r+i >= 0 and r+i < 5 and c+j >= 0 and c+j < 5 and ai.board[r+i][c+j] == ' ':
                                        possible_moves.append([(r+i, c+j), (r, c)])
                ai.opponent_move(random.choice(possible_moves))

            # update the game variables
            turn += 1
            turn %= 2
            if debug:
                ai.print_board()

        # ai.print_board()
        if debug:
            if ai.game_value(ai.board) == 1:
                print("AI wins! Game over.")
            else:
                print("Random player wins! Game over.")
        return ai.game_value(ai.board)

def main():
    test_teeko2_player = TestTeeko2Player()
    test_teeko2_player.test_gameplay()

if __name__ == "__main__":
    main()
