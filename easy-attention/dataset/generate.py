import jax
import jax.numpy as jnp
import sys
import json

def choose(rng, lst):
  [idx] = jax.random.randint(rng, shape=[1], minval=0, maxval=len(lst))
  return lst[idx]

def mknums(rng):
  [amt] = jax.random.randint(rng, shape=[1], minval=3, maxval=9)
  nums = jax.random.randint(rng, shape=[amt], minval=1, maxval=10)
  return nums

def dn(nums):
  return " ".join(str(n) for n in nums)

def among(rng):
  return choose(rng, ["number:", "number among:", "number in the list:", "number in this list:", "of these numbers:"])

def display(rng):
  return choose(rng, [
    "Return", "Return the",
    "Display", "Display the",
    "Show", "Show the",
    "Find", "Find the", "Find and show", "Find and show the",
    "Determine", "Determine the", "Determine which is the",
  ])

def largest(rng):
  d, v, a, n = jax.random.split(rng, 4)
  nums = mknums(n)
  q = f"{display(d)} {choose(v, ['largest','biggest','maximum'])} {among(a)} {dn(nums)}"
  ans = max(nums)
  return q, str(ans)

def smallest(rng):
  d, v, a, n = jax.random.split(rng, 4)
  nums = mknums(n)
  q = f"{display(d)} {choose(v, ['smallest','lowest','minimum'])} {among(a)} {dn(nums)}"
  ans = min(nums)
  return q, str(ans)

def first(rng):
  d, v, a, n = jax.random.split(rng, 4)
  nums = mknums(n)
  q = f"{display(d)} {choose(v, ['first'])} {among(a)} {dn(nums)}"
  ans = nums[0]
  return q, str(ans)

def last(rng):
  d, v, a, n = jax.random.split(rng, 4)
  nums = mknums(n)
  q = f"{display(d)} {choose(v, ['last'])} {among(a)} {dn(nums)}"
  ans = nums[-1]
  return q, str(ans)

def second(rng):
  d, v, a, n = jax.random.split(rng, 4)
  nums = mknums(n)
  q = f"{display(d)} {choose(v, ['second'])} {among(a)} {dn(nums)}"
  ans = nums[1]
  return q, str(ans)

def second_last(rng):
  d, v, a, n = jax.random.split(rng, 4)
  nums = mknums(n)
  q = f"{display(d)} {choose(v, ['second last','penultimate'])} {among(a)} {dn(nums)}"
  ans = nums[-2]
  return q, str(ans)

fs = [
  largest,
  smallest,
#  first,
#  last,
#  second,
#  second_last,
]

def make_training_example(rng1, rng2):
  [which] = jax.random.randint(rng1, shape=[1], minval=0, maxval=len(fs))
  return fs[which](rng2)

def do_1k_examples(rng_key, total_idx):
  rng_key, *rngs = jax.random.split(rng_key, 1+2048)
  for idx in range(0, 2048, 2):
    q, ans = make_training_example(rngs[idx], rngs[idx+1])
    #print(total_idx, idx, q, ans, file=sys.stderr)
    total_idx += 1
#   print(q)
    l = json.dumps(dict(question=q, solution=ans))
    print(l)
  return rng_key, total_idx

rng_key = jax.random.PRNGKey(12)
total_idx = 0
while True:
  rng_key, total_idx = do_1k_examples(rng_key, total_idx)
  print(total_idx, file=sys.stderr)

"""
* {Return,Display,Show} the {largest,biggest,maximum} number: 1 5 3 5 7 2
*                       the {sum,addition} of the numbers:
*                       the {average,mean}
*                       the {mode,most frequent value}
*                       the {median}
*                       the {difference between the largest and smallest}
*                       the {largest,biggest,maximum} {even,odd} number
*                       the {smallest,lowest,minimum} {even,odd} number
*                       the {count of even numbers}
*                       the {count of odd numbers}
*                       the {count of numbers} {above,below} {2,3,4,5,6,7,8}
*                       the {count of numbers} between {1-8} and {2-9}
"""
