"""Continuation must not repeat the original 100 batches or accept real regression."""
import json,math,random,unittest
from pathlib import Path
from sft_v5_continue.runtime import skill_gates,lr_for,loss_retained

class ContinuationTests(unittest.TestCase):
 def test_skip_consumed_examples_without_omitting_tail(self):
  n=63896;order=list(range(n));random.Random(20261018).shuffle(order)
  prior=order[:100*32];remaining=[]
  for step in range(101,math.ceil(n/32)+1):remaining.extend(order[(step-1)*32:step*32])
  self.assertEqual(len(remaining),60696);self.assertFalse(set(prior)&set(remaining));self.assertEqual(prior+remaining,order)
 def test_small_absolute_loss_change_and_real_regression(self):
  def report(value):return {'complete':True,'scores':{'conversation':{'tokens':431,'mean_loss':value}}}
  self.assertTrue(loss_retained(report(.003313),report(.003785),1.05,.01))
  self.assertFalse(loss_retained(report(.003313),report(.03),1.05,.01))
  self.assertFalse(loss_retained(report(.6),report(.7),1.05,.01))
 def test_real_math_regression_still_blocks_selection(self):
  cfg={'pilot_policy':{'natural_math_max_drop':4,'dev_max_extra_truncation':2,'family_loss_max_ratio':1.05,'family_loss_absolute_allowance':.01}}
  d={'n':128,'truncated':0};a={'n':143,'correct':71};b={'n':143,'correct':60};loss={'complete':True,'scores':{'code':{'tokens':10,'mean_loss':.6}}}
  self.assertFalse(skill_gates(d,d,a,b,loss,loss,cfg)['natural_math_retained'])
 def test_restart_learning_rate_ramp(self):
  cfg={'learning_rate':7.5e-7,'warmup_steps':50,'resume_step':100,'resume_warmup_steps':20}
  self.assertLess(lr_for(101,1997,cfg),lr_for(120,1997,cfg));self.assertLessEqual(lr_for(120,1997,cfg),cfg['learning_rate'])

if __name__=='__main__':unittest.main()
