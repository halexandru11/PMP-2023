import scipy.stats as stats


# Punctul a)
def approx():
    N = 10000
    # generate 10000 iterations
    x = stats.geom.rvs(0.3, size=N)
    y = stats.geom.rvs(0.5, size=N)

    # this is the condition
    smaller = x > y**2

    # this is the wanted result: P(X > Y^2) = fav_cases/all_cases
    is_true = smaller.sum() / N

    return is_true


# Punctul b)
def statistics():
    # make 30 iterations
    k = 30
    p = [approx() for _ in range(k)]

    # compute mean
    mean = 0
    for x in p:
        mean += x
    mean /= k

    # compute standart deviation
    std = 0
    for x in p:
        std += (x - mean) ** 2
    std /= k

    return mean, std


if __name__ == "__main__":
    is_true = approx()
    print("P(X > Y^2) = ", is_true)

    mean, std = statistics()
    print()
    print(f"Mean:               {mean}")
    print(f"Standard Deviation: {std}")
