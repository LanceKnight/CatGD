from torch_geometric.nn import GCN as Encoder
import torch


class GCNet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_layers,
                 out_channels,
                 dropout = 0.0,
                 act = 'relu',
                 act_first = False,
                 act_kwargs = None,
                 norm = None,
                 norm_kwargs = None,
                 jk = None,
                 **kwargs):

        super(GCNet, self).__init__()
        self.encoder = Encoder(
                 in_channels,
                 hidden_channels,
                 num_layers,
                 out_channels,
                 dropout = 0.0,
                 act = 'relu',
                 act_first = False,
                 act_kwargs = None,
                 norm = None,
                 norm_kwargs = None,
                 jk = None,
                 **kwargs)

    def forward(self, batch_data):


        batch_data.z = batch_data.x.squeeze()
        graph_embedding = self.encoder(batch_data)

        return graph_embedding

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model arguments to the parent parser
        :param parent_parser: parent parser for adding arguments
        :return: parent parser with added arguments
        """
        parser = parent_parser.add_argument_group("GCNet")

        # Add specific model arguments below
        # E.g., parser.add_argument('--GCN_arguments', type=int,
        # default=12)
        parser.add_argument('--in_channels', type=int, default=6,)
        parser.add_argument('--hidden_channels', type=int, default=3, )
        parser.add_argument('--out_channels', type=int, default=6, )
        parser.add_argument('--num_layers', type=int, default=3, )



        return parent_parser