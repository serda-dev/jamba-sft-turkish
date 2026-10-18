"""Critical gates and accumulation weighting before a paid training job."""
import unittest
from sft_v5.runtime import skill_gates,lr_for
from sft_v5.data import verify

class TrainingChecks(unittest.TestCase):
 def test_natural_math_regression_blocks_even_with_format_gain(self):
  c={'pilot_policy':{'natural_math_max_drop':4,'dev_max_extra_truncation':2,'family_loss_max_ratio':1.05}}
  b={'n':128,'correct':2,'truncated':0};d={'n':128,'correct':100,'truncated':0};m={'n':143,'correct':60};bad={'n':143,'correct':55};loss={'complete':True,'scores':{'code':{'tokens':10,'mean_loss':1.}}}
  self.assertFalse(all(skill_gates(b,d,m,bad,loss,loss,c).values()))
 def test_code_loss_regression_blocks(self):
  c={'pilot_policy':{'natural_math_max_drop':4,'dev_max_extra_truncation':2,'family_loss_max_ratio':1.05}};d={'n':128,'correct':20,'truncated':0};m={'n':143,'correct':60};a={'complete':True,'scores':{'code':{'tokens':10,'mean_loss':1.}}};b={'complete':True,'scores':{'code':{'tokens':10,'mean_loss':1.06}}}
  self.assertFalse(skill_gates(d,d,m,m,a,b,c)['family_losses_retained'])
 def test_accumulation_matches_combined_token_mean(self):
  import torch
  p=torch.tensor(2.,requires_grad=True);(((p-1)**2)*1/4).backward();(((p+1)**2)*3/4).backward();actual=p.grad.item()
  q=torch.tensor(2.,requires_grad=True);torch.stack([(q-1)**2]+[(q+1)**2]*3).mean().backward();self.assertEqual(actual,q.grad.item())
 def test_math_scoring_and_warmup(self):
  row={'family':'natural_math','gold':'42'};self.assertTrue(verify(row,'6 × 7 = 42.\n#### 42')['correct']);self.assertFalse(verify(row,'#### 41')['correct'])
  cfg={'learning_rate':7.5e-7,'warmup_steps':50};self.assertLess(lr_for(1,2000,cfg),lr_for(50,2000,cfg));self.assertLess(lr_for(2000,2000,cfg),lr_for(50,2000,cfg))

if __name__=='__main__':unittest.main()
