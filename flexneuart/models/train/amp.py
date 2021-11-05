#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

#
# Mixed precision helper classes/functions
#


"""
  A dummy autocast class allowing for easy enabling/disabling of AMP.
"""
class DummyAutoCast:

    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        pass


"""
  A dummy gradient scaler class allowing for easy enabling/disabling of AMP.
"""
class DummyGradScaler:
    # just return the unscaled loss
    def scale(self, loss):
        return loss

    # just do the optimizer step without any changes.
    def step(self, optimizer):
        optimizer.step()

    # clearly nothing to update here
    def update(self):
        pass


"""
  There's "enabled" flag in autocast and GradScaler, however, 
  it is not clear if using it has no effect (as we need). 
  To ensure autocast is not involved at all, we introduce these dummy classes.

  As an additional benefit, this,  permits using older version of 
  Pytorch that have no built-in amp. 
  
  Thus, if one does not have amp or you does not want to use it,
  we do not have to use different training code.

"""
def get_amp_processors(enabled):
    if enabled:
      from torch.cuda.amp import autocast, GradScaler
      return autocast, GradScaler()
    else:
      return DummyAutoCast, DummyGradScaler()

