

# coin_set = [50, 20, 10]
coins = {50: 0, 20: 0, 10: 0, 5: 0}
def make_change(change, coins=coins):
    for c in coins:
        coin_num = int(change/c)
        coins[c] = coin_num
        change -= coin_num * c
    return coins


if __name__ == "__main__":
    print(make_change(83))
