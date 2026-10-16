from typing import Tuple
import numpy as np
from numba import njit

# Every compiled function in the project lives in this one module, on purpose: numba's
# on-disk cache only notices edits to the file of the function being called, so a helper
# imported from elsewhere could change underneath a cached caller and go on running stale.

# Fibonacci hashing: multiplying by 2^64 / phi spreads even consecutive running totals
# evenly over a table, and folding the high half down puts that spread in the low bits
GOLDEN = np.uint64(0x9E3779B97F4A7C15)

# a slot no entry has claimed; positions and offsets are never negative, so it cannot clash
EMPTY = -1

# columns of the scan's table: a running total, and how many pending people it was reached at
TOTAL = 0
AT = 1

# columns of the cost cache: the members' hash, where they sit in the arena, how many there
# are, and what the group costs
HASH = 0
START = 1
SIZE = 2
COST = 3

# capacity of the cost cache: half as many groups as slots, and members in the arena. Sized
# so that a whole neighbourhood of instancia_splitwise_200 fits; larger ones may start over
COST_SLOTS = 1 << 16
ARENA_SIZE = 1 << 21

# counters of the cost cache, kept in an array so that a helper can bump them in place
ENTRIES = 0
ARENA = 1
EPOCH = 2

@njit(cache=True, inline='always')
def _mix(value: int) -> np.uint64:
    mixed = np.uint64(value) * GOLDEN

    return mixed ^ (mixed >> np.uint64(32))

@njit(cache=True)
def _table(people: int) -> np.ndarray:
    """An empty scan table, sized to stay at most a quarter full.

    Open addressing with linear probing over a power-of-two capacity. A quarter is the bound
    because a neighbourhood fills it twice over: the scan up to i, and a tail on top of it.
    """
    capacity = 1

    while capacity < 4 * (people + 1):
        capacity <<= 1

    table = np.full((capacity, 2), EMPTY, np.int64)

    # the empty prefix, which a group summing to zero from the very start repeats
    table[_slot(table, 0), TOTAL] = 0
    table[_slot(table, 0), AT] = 0

    return table

@njit(cache=True, inline='always')
def _slot(table: np.ndarray, total: int) -> int:
    """The slot holding `total`, or the empty one it would go in."""
    mask = table.shape[0] - 1
    slot = np.int64(_mix(total) & np.uint64(mask))

    while table[slot, AT] != EMPTY and table[slot, TOTAL] != total:
        slot = (slot + 1) & mask

    return slot

@njit(cache=True, inline='always')
def _add(
    person: int,
    balances: np.ndarray,
    pending: np.ndarray,
    running: np.ndarray,
    length: int,
    table: np.ndarray,
    undo: np.ndarray,
    logged: int,
    log: bool,
) -> Tuple[int, int]:
    """Extend the scan by one person: the new length, and how much of `undo` is now in use.

    The state is `pending[:length]`, the people not yet in a closed group, and
    `running[:length + 1]`, the running total after each of them -- all distinct, since a
    repeat closes a group. The table maps each of those totals to where it was reached, and
    is never cleared: an entry counts only while `running` still holds its total at its
    position, so truncating `length` is all it takes to drop every total inside a group.

    A group closed exactly when the length did not grow, and it is
    `pending[new length:length + 1]`.

    With `log`, every slot written is recorded in `undo` first, so that a tail scanned on
    top of a shared prefix can be rolled back for the next one.
    """
    total = running[length] + balances[person]
    pending[length] = person

    slot = _slot(table, total)
    at = table[slot, AT]

    # the balances between two positions sum to zero exactly when the running total is the
    # same at both -- the value they share is irrelevant, only that it repeats
    if at != EMPTY and at <= length and running[at] == total:
        return at, logged

    if log:
        undo[logged, 0] = slot
        undo[logged, 1] = table[slot, TOTAL]
        undo[logged, 2] = at
        logged += 1

    length += 1
    running[length] = total
    table[slot, TOTAL] = total
    table[slot, AT] = length

    return length, logged

@njit(cache=True, inline='always')
def _undo(table: np.ndarray, undo: np.ndarray, logged: int) -> None:
    """Roll the table back over every logged write, newest first."""
    for entry in range(logged - 1, -1, -1):
        slot = undo[entry, 0]
        table[slot, TOTAL] = undo[entry, 1]
        table[slot, AT] = undo[entry, 2]

@njit(
    'Tuple((int64[::1], int64[::1]))(int64[::1], int64[::1])',
    cache=True,
)
def zero_sum_groups(balances: np.ndarray, order: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split `order` into groups whose balances sum to zero, cutting at every repeat.

    Cutting only where the total is zero, which is the special case of repeating the empty
    prefix, would throw away every other repeat. Removing a group changes nothing about the
    totals that follow it, since the group contributed zero, so a single pass suffices.

    Returns the members group after group, and where each group ends among them.
    """
    people = order.size

    pending = np.empty(people, np.int64)
    running = np.zeros(people + 1, np.int64)
    table = _table(people)
    undo = np.empty((0, 3), np.int64)

    members = np.empty(people, np.int64)
    ends = np.empty(people, np.int64)
    filled = 0
    groups = 0
    length = 0

    for person in order:
        closed, _ = _add(person, balances, pending, running, length, table, undo, 0, False)

        if closed <= length:
            size = length + 1 - closed
            members[filled:filled + size] = pending[closed:length + 1]
            filled += size
            ends[groups] = filled
            groups += 1

        length = closed

    return members[:filled], ends[:groups]

@njit(cache=True, inline='always')
def _push(heap: np.ndarray, size: int, value: int) -> None:
    """Push onto a max-heap of `size` entries."""
    position = size

    while position > 0:
        parent = (position - 1) >> 1

        if heap[parent] >= value:
            break

        heap[position] = heap[parent]
        position = parent

    heap[position] = value

@njit(cache=True, inline='always')
def _pop(heap: np.ndarray, size: int) -> int:
    """Drop the top of a max-heap of `size` entries; read heap[0] first."""
    size -= 1
    last = heap[size]
    position = 0

    while True:
        child = 2 * position + 1

        if child >= size:
            break

        if child + 1 < size and heap[child + 1] > heap[child]:
            child += 1

        if heap[child] <= last:
            break

        heap[position] = heap[child]
        position = child

    if size > 0:
        heap[position] = last

    return size

@njit(cache=True)
def _count_largest_first(
    balances: np.ndarray,
    people: np.ndarray,
    lo: int,
    hi: int,
    owes: np.ndarray,
    dues: np.ndarray,
) -> int:
    """How many transactions GreedyAlgorithm.settle_largest_first settles people[lo:hi] in.

    Magnitudes alone decide the count: a pop settles two people rather than one exactly when
    owes equals due, and who is who changes which pair settles, never how many do. So this
    keeps no people, and builds none of the arrays the greedy hands back.
    """
    payers = 0
    receivers = 0

    for index in range(lo, hi):
        balance = balances[people[index]]

        if balance < 0:
            _push(owes, payers, -balance)
            payers += 1
        elif balance > 0:
            _push(dues, receivers, balance)
            receivers += 1

    count = 0

    while payers > 0 and receivers > 0:
        owed = owes[0]
        payers = _pop(owes, payers)
        due = dues[0]
        receivers = _pop(dues, receivers)

        count += 1

        if owed > due:
            _push(owes, payers, owed - due)
            payers += 1
        elif owed < due:
            _push(dues, receivers, due - owed)
            receivers += 1

    return count

@njit(cache=True, inline='always')
def _cost(
    balances: np.ndarray,
    people: np.ndarray,
    lo: int,
    hi: int,
    key: int,
    costs: np.ndarray,
    arena: np.ndarray,
    counters: np.ndarray,
    stamps: np.ndarray,
    owes: np.ndarray,
    dues: np.ndarray,
) -> int:
    """What the group `people[lo:hi]` costs, from the cache where possible.

    A group's cost depends on its members as a set, so the cache is keyed by one: `key` is
    the xor of the members' random keys, which the scan keeps as a running prefix and so
    costs nothing per group. A hash is no proof of membership, though, and the fitness must
    not answer to a collision -- so a hit is confirmed member by member against the stored
    group, by stamping the members of this one first. Confirming is a single pass, where the
    greedy it stands in for is a heap pass.

    Fixed in size, and emptied whenever it fills: a memo can forget at any point without
    changing a single answer. Growing it instead would mean handing new arrays back to the
    caller, and merely reassigning them inside the search loop measured 2.6 times slower
    over a whole neighbourhood, for the reference counting it brings into the hottest loop.
    """
    size = hi - lo
    mask = costs.shape[0] - 1
    slot = np.int64(_mix(key) & np.uint64(mask))

    counters[EPOCH] += 1
    epoch = counters[EPOCH]

    for index in range(lo, hi):
        stamps[people[index]] = epoch

    while costs[slot, START] != EMPTY:
        if costs[slot, HASH] == key and costs[slot, SIZE] == size:
            start = costs[slot, START]
            same = True

            for index in range(start, start + size):
                if stamps[arena[index]] != epoch:
                    same = False

                    break

            if same:
                return costs[slot, COST]

        slot = (slot + 1) & mask

    cost = _count_largest_first(balances, people, lo, hi, owes, dues)

    # at most half full, so that a probe for a group never seen ends quickly
    if counters[ARENA] + size > arena.size or 2 * (counters[ENTRIES] + 1) > costs.shape[0]:
        costs[:, START] = EMPTY
        counters[ENTRIES] = 0
        counters[ARENA] = 0
        slot = np.int64(_mix(key) & np.uint64(mask))

    start = counters[ARENA]
    arena[start:start + size] = people[lo:hi]
    counters[ARENA] += size

    costs[slot, HASH] = key
    costs[slot, START] = start
    costs[slot, SIZE] = size
    costs[slot, COST] = cost
    counters[ENTRIES] += 1

    return cost

@njit(
    'UniTuple(int64, 3)(int64[::1], int64[::1], int64, int64, int64[::1], boolean)',
    cache=True,
)
def best_swap(
    balances: np.ndarray,
    order: np.ndarray,
    fitness: int,
    pairs: int,
    keys: np.ndarray,
    first: bool,
) -> Tuple[int, int, int]:
    """The swap (i, j) that improves most on `fitness`, and what it scores; i is -1 if none.

    With `first`, the first improving swap instead, in the order (0, 1), (0, 2), ... The
    incumbent seeds the best, so a neighbour has to be strictly better to be kept at all,
    and ties among equally good ones go to the first scanned.

    The scan up to i is the same for every j, so it is kept as a base that advances by one
    person per i, and each tail is scanned on top of a copy of it. The table those share is
    rolled back through `undo` rather than copied, because it is sized for the whole order
    and a tail touches only a few of its slots. `pairs` is what the pairing settled before
    any of this, which no order changes.
    """
    people = order.size

    table = _table(people)
    undo = np.empty((people + 1, 3), np.int64)

    base_pending = np.empty(people, np.int64)
    base_running = np.zeros(people + 1, np.int64)
    base_hashes = np.zeros(people + 1, np.int64)
    base_length = 0
    base_cost = 0

    pending = np.empty(people, np.int64)
    running = np.zeros(people + 1, np.int64)
    hashes = np.zeros(people + 1, np.int64)

    # cached for this neighbourhood alone, not for the whole search, which keeps the call a
    # function of its arguments and nothing else
    costs = np.full((COST_SLOTS, 4), EMPTY, np.int64)
    # never smaller than one group can be, which is every person in the order
    arena = np.empty(max(ARENA_SIZE, people), np.int64)
    counters = np.zeros(3, np.int64)
    stamps = np.zeros(balances.size, np.int64)
    owes = np.empty(people, np.int64)
    dues = np.empty(people, np.int64)

    best_fitness = fitness
    best_i = -1
    best_j = -1

    for i in range(people - 1):
        balance = balances[order[i]]

        for j in range(i + 1, people):
            # equal balances swap to the same fitness, so this neighbour cannot win
            if balances[order[j]] == balance:
                continue

            pending[:base_length] = base_pending[:base_length]
            running[:base_length + 1] = base_running[:base_length + 1]
            hashes[:base_length + 1] = base_hashes[:base_length + 1]
            length = base_length
            cost = base_cost
            logged = 0

            for position in range(i, people):
                if position == i:
                    person = order[j]
                elif position == j:
                    person = order[i]
                else:
                    person = order[position]

                hashes[length + 1] = hashes[length] ^ keys[person]
                closed, logged = _add(
                    person, balances, pending, running, length, table, undo, logged, True
                )

                if closed <= length:
                    cost += _cost(
                        balances, pending, closed, length + 1,
                        hashes[length + 1] ^ hashes[closed],
                        costs, arena, counters, stamps, owes, dues,
                    )

                length = closed

            _undo(table, undo, logged)

            if pairs + cost < best_fitness:
                best_fitness = pairs + cost
                best_i = i
                best_j = j

                if first:
                    return best_i, best_j, best_fitness

        person = order[i]
        base_hashes[base_length + 1] = base_hashes[base_length] ^ keys[person]
        closed, _ = _add(
            person, balances, base_pending, base_running, base_length, table, undo, 0, False
        )

        if closed <= base_length:
            base_cost += _cost(
                balances, base_pending, closed, base_length + 1,
                base_hashes[base_length + 1] ^ base_hashes[closed],
                costs, arena, counters, stamps, owes, dues,
            )

        base_length = closed

    return best_i, best_j, best_fitness
