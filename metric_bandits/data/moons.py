from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

n_samples = 1000
noise = 0.2

X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
moons = (X, y)  # this is the actual data
