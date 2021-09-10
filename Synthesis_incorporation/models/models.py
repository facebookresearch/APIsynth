import torch
import torch as T
from torch import nn
EMBEDDING_SIZE = 150
SHAPE_EMBEDDING_SIZE = 6

class pycoder_parameters:

    ''' Core Fuzzing Parameters '''
    NUM_FUZZ_PER_API= 100000 #000
    NUM_TEST_FUZZ = 2
    FLOAT_TENSOR = False #We either generate float or integer tensors
    UNIT_TEST = False
    COMPOSITE = True

    ''' Fuzzing Detailed Parameters '''
    MAX_TENSOR_DIMENSIONS = 3 #how many rows, columns, etc.
    MIN_VAL_PER_DIMENSION = 1 # e.g., min number of rows, columns, etc.
    MAX_VAL_PER_DIMENSION = 5 # e.g., max number of rows, columns, etc.

    #So far limiting to integers
    MIN_TENSOR_VALUE = 1
    MAX_TENSOR_VALUE = 15


    ''' Embedding Parameters '''
    EMBEDDING_NOISE_LEVEL = 0 #0 noise by default
    EMBEDDING_SIZE = 150
    SHAPE_EMBEDDING_SIZE = 6


    data_type = 'float' if FLOAT_TENSOR is  True else 'integer'
    model_type = 'Composite_' if COMPOSITE is  True else 'Single_'
    file_name = str(model_type) + str(NUM_FUZZ_PER_API) + '_' + data_type
    fuzzing   = file_name + '.pt'
    embedding = file_name + '.embedding' + '.pt',
    classification = file_name + '.model_result' + '.pt'
    train_valid_test = file_name + 'train_valid_test.pt'

    def setNoiseLevel(self, noise):
        self.EMBEDDING_NOISE_LEVEL = noise
        self.embedding = self.file_name + '.embedding' + '_' + str(self.EMBEDDING_NOISE_LEVEL) + '.pt'

    def getEmbeddingFile(self):
        return(self.file_name + '.embedding' + '_' + str(self.EMBEDDING_NOISE_LEVEL) + '.pt')

    def getVisulizationFile(self):
        return(self.file_name + '.embedding' + '_' + str(self.EMBEDDING_NOISE_LEVEL) + '_' +  'tSNE.pt')

class Net(torch.nn.Module):
    def __init__(self, settings, len_api):
        super(Net, self).__init__()

        first_layer_size = settings.model.embedding_size
        if settings.model.use_shape_encoding:
            first_layer_size += settings.model.shape_embedding_size
        if settings.model.use_type_encoding:
            first_layer_size += 2
        self.hid1 = torch.nn.Linear(4*(first_layer_size+1), 500)
        self.hid2 = torch.nn.Linear(500, 250)
        self.hid3 = torch.nn.Linear(250, 100)
        self.oupt = torch.nn.Linear(100, len_api)

        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.oupt.weight)
        torch.nn.init.zeros_(self.oupt.bias)

        torch.nn.Dropout(p=0.2)

    def forward(self, x):
        z1 = torch.tanh(self.hid1(x))
        z2 = torch.tanh(self.hid2(z1))
        z3 = torch.tanh(self.hid3(z2))
        z = self.oupt(z3)  # no softmax: CrossEntropyLoss()
        return (z, z3, z2, z1)


class FFNet(T.nn.Module):
    def __init__(self):
        super(FFNet, self).__init__()
        NOISE = 0
        f = pycoder_parameters()
        f.setNoiseLevel(NOISE)
        f.embedding = f.getEmbeddingFile()
        print(f.embedding)
        print(f.SHAPE_EMBEDDING_SIZE)

        self.hid1 = T.nn.Linear(4*(f.EMBEDDING_SIZE+f.SHAPE_EMBEDDING_SIZE+1+2), 500)
        self.hid2 = T.nn.Linear(500, 250)
        self.hid3 = T.nn.Linear(250, 100)
        # self.oupt = T.nn.Linear(100, len(api2indx))
        self.oupt = T.nn.Linear(100, 33)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

        T.nn.Dropout(p=0.2)


    def forward(self, x):
        z1 = T.tanh(self.hid1(x))
        z2 = T.tanh(self.hid2(z1))
        z3 = T.tanh(self.hid3(z2))
        z = self.oupt(z3)  # no softmax: CrossEntropyLoss()
        return (z, z3, z2, z1)


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNNModel, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim*2, output_size)


    def forward(self, x):
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        out1 = out.contiguous().view(-1, self.hidden_dim*2)
        out1 = self.fc(out1)

        return out1, hidden, out

    def init_hidden(self, batch_size):
        device = T.device("cpu")
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim).to(device)
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden
