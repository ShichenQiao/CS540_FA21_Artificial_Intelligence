import heapq


# return the sum of Manhattan distance of each tile to its goal position
def get_heuristic(state):
    h = 0
    for i in range(3):                          # row number in state
        for j in range(3):                      # col number in state
            num = state[i * 3 + j]              # number at a row i col j in the given state
            if num != 0:                        # count towards the heuristic for all numbers but 0
                x = (num - 1) % 3               # col number in goal
                y = int((num - 1) / 3)          # row number in goal
                h += abs(y - i) + abs(x - j)    # Manhattan distance of the current tile
    return h


# given a state of the puzzle print to the console all of the possible successor states
def get_succ(state):
    rst = []
    index = state.index(0)  # get index of the empty place
    # up
    if index - 3 in range(9):
        temp = state.copy()
        temp[index] = state[index - 3]
        temp[index - 3] = 0
        rst.append(temp)

    # down
    if index + 3 in range(9):
        temp = state.copy()
        temp[index] = state[index + 3]
        temp[index + 3] = 0
        rst.append(temp)

    # left
    if (index % 3 != 0) and (index - 1 in range(9)):
        temp = state.copy()
        temp[index] = state[index - 1]
        temp[index - 1] = 0
        rst.append(temp)

    # right
    if (index % 3 != 2) and (index + 1 in range(9)):
        temp = state.copy()
        temp[index] = state[index + 1]
        temp[index + 1] = 0
        rst.append(temp)

    return sorted(rst)           # if consider the state to be a nine-digit integer, sorted in ascending order


# format and print the successor states of the given state, as well as their heuristic values
def print_succ(state):
    rst = get_succ(state)
    for succ in rst:
        print(succ, "h=%d" % get_heuristic(succ))


# solve the puzzle
def solve(state):
    goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    pq = []                             # priority queue, pop out state with lowest g + h
    close = []                          # list to track popped states
    max_len_q = 0                       # max length of pq
    ans = None                          # reference to the node containing the goal state
    h = get_heuristic(state)
    heapq.heappush(pq, (h, state, (0, h, -1)))        # push initial state to pq, it's parent index is -1
    while len(pq) != 0:                 # since we assumed the given puzzle is solvable, loop until find goad
        # record max length of the pq
        l = len(pq)
        if l > max_len_q:
            max_len_q = l
        # pop the state with lowest g + h in the pq and store it into the closed list
        n = heapq.heappop(pq)
        close.append(n)
        if n[1] == goal:                # check if the popped state is the goal state
            ans = n
            break                       # record reference and exit loop if yes
        # if goal state not reached, process the popped node n
        succs = get_succ(n[1])          # get n's successors
        g = n[2][0] + 1                 # g(n') = g(n) + 1
        for succ in succs:                      # go through all n' for n
            if succ not in [x[1] for x in close]:
                h = get_heuristic(succ)         # h(n')
                processed = False
                # if n' already exist in the pq, push n' only if g(n') is lower for the new n'
                for i in range(len(pq)):
                    if (succ in pq[i]) and (pq[i][2][0] > g):
                        heapq.heappush(pq, (g + h, succ, (g, h, close.index(n))))       # parent index is index in close
                        processed = True
                # if n' is not in the pq just push it to pq
                if not processed:
                    heapq.heappush(pq, (g + h, succ, (g, h, close.index(n))))
    index = close.index(ans)            # find the index of the goal state in close
    moves = []                          # record moves taken from the initial state to the goal state
    while index != -1:                  # trace back until reaches the initial state
        moves.insert(0, close[index][1])        # record each move in the order from beginning to the end
        index = close[index][2][2]              # trance back to parent
    cnt = 0                             # count the number of moves
    for move in moves:                  # format and print the moves
        print(move, "h=%d" % get_heuristic(move), "moves:", cnt)
        cnt += 1
    print("Max queue length:", max_len_q)       # print the max length of pq during the procedure
