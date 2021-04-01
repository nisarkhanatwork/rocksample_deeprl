import numpy as np

# discount rewards used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def discount_rewards(r, gamma, y_pred):
  """ take 1D float array of rewards and compute discounted reward """
  r = np.array(r)
  y_pred = np.array(y_pred)
  discounted_r = np.zeros_like(r).astype(np.float64)
  running_add = 0
  for t in range(0, r.size):
    if(t == (r.size - 1)):
        y_pred[t] = 0
    discounted_r[t] = r[t] + gamma * y_pred[t]

  return discounted_r
