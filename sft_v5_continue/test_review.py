import unittest
from sft_v5_continue.review_policy import eligible,pause_ready

class ReviewTests(unittest.TestCase):
 def test_pause_only_after_last_update_and_complete_development(self):
  self.assertFalse(pause_ready({'tag':'dev_1600','n':128}))
  self.assertFalse(pause_ready({'tag':'dev_1997','n':127}))
  self.assertTrue(pause_ready({'tag':'dev_1997','n':128}))
 def test_generated_conversation_accuracy_cannot_be_replaced_by_loss(self):
  cfg={'pilot_policy':{'natural_math_max_drop':4,'dev_max_extra_truncation':2,'family_loss_max_ratio':1.05,'family_loss_absolute_allowance':.01}}
  d={'n':128,'truncated':0};math={'n':143,'correct':69};base_math={'n':143,'correct':71}
  base={'complete':True,'scores':{'code':{'tokens':10,'mean_loss':.6},'conversation':{'tokens':431,'mean_loss':.0033}}}
  new={'complete':True,'scores':{'code':{'tokens':10,'mean_loss':.58},'conversation':{'tokens':431,'mean_loss':.016}}}
  conversation={'n':15,'correct':15,'truncated':0}
  gates=eligible(d,d,base_math,math,base,new,conversation,conversation,cfg);self.assertTrue(all(gates.values()))
  bad={**conversation,'correct':14};self.assertFalse(eligible(d,d,base_math,math,base,new,conversation,bad,cfg)['conversation_generated_retained'])
 def test_real_math_drop_still_excludes_candidate(self):
  cfg={'pilot_policy':{'natural_math_max_drop':4,'dev_max_extra_truncation':2,'family_loss_max_ratio':1.05,'family_loss_absolute_allowance':.01}}
  d={'n':128,'truncated':0};loss={'complete':True,'scores':{'code':{'tokens':10,'mean_loss':.6}}};c={'n':15,'correct':15,'truncated':0}
  self.assertFalse(eligible(d,d,{'n':143,'correct':71},{'n':143,'correct':60},loss,loss,c,c,cfg)['math_retained'])

if __name__=='__main__':unittest.main()
