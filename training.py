from blackjack import *
from neural_network import *
from ai_player import *
import datetime

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = torch.load('model_init.pt')
for i in range(1000):
    log = ai_game(model,loop_num=500)
    model = ai_update(model,log,epochs=2000)
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"),'progress:',i/10,'%')
torch.save(model, 'model.pt')
log = ai_game(model,loop_num=100)
get_top_winners(log, n=2)