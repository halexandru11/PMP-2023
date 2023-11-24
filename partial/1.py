import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx

# Ex 1.1


def ex1():
    # numarul de victorii pt fiecare jucator
    wins_p0 = 0
    wins_p1 = 0

    # simulez 20000 de jocuri
    for _epoch in range(20000):
        # Pun 1 la jucatorul care incepe primul
        first_player = 0
        if random.random() < 0.5:  # [0, 1]
            first_player = 1

        # initializez numarul de steme din prima runda. valoarea va fi suprascrisa
        n = 0
        if first_player == 1:
            # jucatorul p0 incepe
            # jucatorul p0 are stema cu sansa de 1/3. Fac o singura aruncare
            n = stats.binom.rvs(1, 1 / 3)  # n va fi 0 sau 1
        else:
            # jucatorul p1 incepe
            # jucatorul p1 are stema cu sansa 1/2. Fac o singura aruncare
            n = stats.binom.rvs(1, 1 / 2)  # n va fi 0 sau 1

        m = 0
        for i in range(n + 1):
            if first_player == 0:
                # jucatorul p0 arunca in a doua runda
                # simulez o aruncare
                m += stats.binom.rvs(1, 1 / 3)
            else:
                # jucatorul p1 arunca in a doua runda
                # simulez o aruncare
                m += stats.binom.rvs(1, 1 / 2)

        if n >= m:
            if first_player == 1:  # p0 a castigat
                wins_p0 += 1
            else:  # p1 a castigat
                wins_p1 += 1
        else:
            if first_player == 1:  # p1 a castigat
                wins_p1 += 1
            else:  # p0 a castigat
                wins_p0 += 1

    if wins_p0 > wins_p1:
        print(f"jucatorul p0 a castigat cu scorul: {wins_p0} la {wins_p1}")
    elif wins_p1 > wins_p0:
        print(f"jucatorul p1 a castigat cu scorul: {wins_p1} la {wins_p0}")
    else:
        print(f"Egalitate: {wins_p0} la {wins_p1}")


# Ex 1.2
def ex2():
    model = BayesianNetwork(
        [("FirstPlayer", "n"), ("n", "m"), ("FirstPlayer", "m")]
    )
    cdp_first_player = TabularCPD(
        variable="FirstPlayer", variable_card=2, values=[[0.5], [0.5]]
    )
    # notez cu fp - FirstPlayer
    cdp_n = TabularCPD(
        variable="n",
        variable_card=2,
        values=[
            [2 / 3, 1 / 2],  # P(n=0|fp=p0)  P(n=0|fp=p1)
            [1 / 3, 1 / 2],  # P(n=1|fp=p0)  P(n=1|fp=p1)
        ],
        evidence=["FirstPlayer"],
        evidence_card=[2],
    )
    cdp_m = TabularCPD(
        variable="m",
        variable_card=2,
        values=[
            [
                2 / 3,
                1 / 3,
                1 / 2,
                1 / 2,
            ],  # P(m=0|n=0,fp=0) P(m=0|n=0,fp=1) P(m=0|n=1,fp=0) P(m=0|n=1,fp=1)
            [
                1 / 3,
                2 / 3,
                1 / 2,
                1 / 2,
            ],  # P(m=1|n=0,fp=0) P(m=1|n=0,fp=1) P(m=1|n=1,fp=0) P(m=1|n=1,fp=1)
        ],
        evidence=["n", "FirstPlayer"],
        evidence_card=[2, 2],
    )

    model.add_cpds(cdp_first_player, cdp_n, cdp_m)

    # verificam modelul
    assert model.check_model()

    pos = nx.circular_layout(model)
    nx.draw(
        model,
        pos=pos,
        with_labels=True,
        node_size=4000,
        font_weight="bold",
        node_color="skyblue",
    )
    plt.show()

    return model


def ex3(model):
    infer = VariableElimination(model)

    result = infer.query(variables=["n"], evidence={"m": 0})
    values = result.values
    # values[0] = P(n=0 | m=0)
    # values[1] = P(n=1 | m=0)
    if values[0] > values[1]:
        print(f"Este mai probabil sa nu pice stema ({values[0]})")
    else:
        print(f"Este mai probabil sa pice stema ({values[1]})")


model = ex2()
ex3(model)
