from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx


# EX 1

model = BayesianNetwork(
    [("Cutremur", "Incendiu"), ("Incendiu", "Alarma"), ("Cutremur", "Alarma")]
)

cpd_cutremur = TabularCPD(
    variable="Cutremur", variable_card=2, values=[[0.9995], [0.0005]]
)
# Cutremur = 0 nu a fost cutremur, Cutremur = 1 a fost cutremur

cpd_incendiu = TabularCPD(
    variable="Incendiu",
    variable_card=2,
    values=[
        [0.99, 0.01],
        [0.03, 0.97],
    ],
    evidence=["Cutremur"],
    evidence_card=[2],
)

cpd_alarma = TabularCPD(
    variable="Alarma",
    variable_card=2,
    values=[
        [0.9999, 0.02, 0.95, 0.98],
        [0.0001, 0.98, 0.05, 0.02],
    ],
    evidence=["Incendiu", "Cutremur"],
    evidence_card=[2, 2],
)


# Adaugam CPD-urile la reteaua bayesiana
model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarma)

# Verificam consistenta retelei bayesiene
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


# EX 2

# Cream un obiect pentru eliminarea variabilelor
infer = VariableElimination(model)

# Calculam probabilitatea P(Cutremur=1 | Alarma=1)
result = infer.query(variables=["Cutremur"], evidence={"Alarma": 1})
print(result)


# EX 3

# Calculam probabilitatea P(Incendiu=1, Alarma=0)
result = infer.query(variables=["Incendiu"], evidence={"Alarma": 0})
print(result.values)
