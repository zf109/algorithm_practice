def _get_temp(start, end):
    all_ = {"A", "B", "C"}
    return all_.difference({start, end}).pop()

def move_layer(tower, start="A", end="C"):
    tower[end].insert(0, tower[start].pop(0))
    print(tower)
    return tower

def move_tower(tower, n, start, end):
    if n == 1:
        tower = move_layer(tower, start, end)
        return tower
    tmp_loc = _get_temp(start, end)
    tower = move_tower(tower, n-1, start=start, end=tmp_loc)
    tower = move_layer(tower, start, end)
    tower = move_tower(tower, n-1, start=tmp_loc, end=end)
    return tower

if __name__ == "__main__":

    tower_size = 4
    start = "A"
    end = "C"

    tower = {"A": [], "B": [], "C": []}
    tower[start] = [x + 1 for x in range(tower_size)]
    tower = move_tower(tower, tower_size, start, end)
