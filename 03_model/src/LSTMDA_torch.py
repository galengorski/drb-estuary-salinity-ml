import torch
import torch.nn as nn
import time

torch.manual_seed(10)


# Simple LSTM made from scratch
class LSTMDA(nn.Module):
    def __init__(self, input_dim, hidden_dim, recur_dropout = 0, dropout = 0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.weight_ih = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))
        self.init_weights()
        
        self.dropout = nn.Dropout(dropout)
        self.recur_dropout = nn.Dropout(recur_dropout)
        
        self.dense = nn.Linear(hidden_dim, 1)
    
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        
    def forward(self, x, init_states = None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
        
        x = self.dropout(x)
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * self.recur_dropout(g_t)
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(1))
        hidden_seq = torch.cat(hidden_seq, dim= 1)
        out = self.dense(hidden_seq)
        return out, (h_t, c_t)
    
    def evaluate(self, x_val, y_val):
    # return predictions and loss for dataset
    # load all the data at the same time
    # data_loader = DataLoader(dataset = dataset, batch_size = len(dataset), 
    #    shuffle = False, drop_last = False, pin_memory = False)

        for i, data in enumerate(x_val):
            # input: tensor of shape (batch_size, window_size, input_size)
            input = x_val 
            target = y_val
            #input = input.to(device) 
            #target = target.to(device) 
        
            with torch.no_grad():
                prediction, _ = self(input)
                loss = rmse_masked(target, prediction)
        
        return prediction, loss



def rmse_masked(y_true, y_pred):
    num_y_true = torch.count_nonzero(
        ~torch.isnan(y_true)
    )
    zero_or_error = torch.where(
        torch.isnan(y_true), torch.zeros_like(y_true), y_pred - y_true
    )
    sum_squared_errors = torch.sum(torch.square(zero_or_error))
    rmse_loss = torch.sqrt(sum_squared_errors / num_y_true)
    return rmse_loss


def mse_masked(y_true, y_pred):
    num_y_true = torch.count_nonzero(
        ~torch.isnan(y_true)
    )
    zero_or_error = torch.where(
        torch.isnan(y_true), torch.zeros_like(y_true), y_pred - y_true
    )
    sum_squared_errors = torch.sum(torch.square(zero_or_error))
    mse_loss = sum_squared_errors / num_y_true
    return mse_loss


# def extreme_loss(y_true, y_pred):
#     y_true_or_err = torch.where(
#         torch.isnan(y_true), torch.Tensor([-9999]) , y_true
#     )
#     #-0.0138 is approx river mile 70 scaled by mean and sd of the training set
#     low_zero_or_error = torch.where(
#         (y_true_or_err >= -0.0138) | (y_true_or_err < -999), torch.zeros_like(y_true_or_err), y_pred - y_true_or_err
#         )
#     num_low_y_true = torch.count_nonzero(
#         low_zero_or_error
#         )
#     high_zero_or_error = torch.where(
#         (y_true_or_err < -0.0138) , torch.zeros_like(y_true_or_err), y_pred - y_true_or_err
#         )
#     num_high_y_true = torch.count_nonzero(
#         high_zero_or_error
#         )

#     sum_squared_errors_low = torch.sum(torch.square(low_zero_or_error))
#     loss_low = torch.sqrt(sum_squared_errors_low / num_low_y_true)
    
#     #cube the errors where river mile is high
#     sum_cubed_errors_high = torch.sum(torch.pow(high_zero_or_error,4))
#     loss_high = (sum_cubed_errors_high / num_high_y_true)
    
#     loss_hi_low = loss_low.add(loss_high)
    
#     return loss_hi_low
    


# def rmse_weighted(y_true, y_pred): # weighted by covariance matrix from DA; weights are concatonated onto y_true and need to separate out within function 
#     raise(NotImplementedError)
#     return rmse_loss
  

def fit_torch_model(model, x, y, x_val, y_val, epochs, loss_fn, optimizer):
    running_loss_train = []
    running_loss_val = []

    for i in range(epochs):
        start_time = time.time()
        
        if i == 0:
            out, (h, c) = model(x)
        else:
            out, (h, c) = model(x, (h.detach(), c.detach())) # stateful lstm
            # .detach() because prev h/c are tied to gradients/weights of
            # a different iteration
        loss = loss_fn(y, out)
        running_loss_train.append(loss.item())
        
        val_preds, val_loss = model.evaluate(x_val, y_val)
        running_loss_val.append(val_loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        end_time = time.time()
        loop_time = end_time - start_time
        
        print('Epoch %i/' %(i+1) + str(epochs), flush = True)
        print('[==============================]',
              '{0:.2f}'.format(loop_time) + 's/step',
              '- loss: ' + '{0:.4f}'.format(loss.item()),
              flush = True)
        
    return model, out, running_loss_train, running_loss_val

