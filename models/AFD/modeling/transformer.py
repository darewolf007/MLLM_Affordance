import math
from typing import Tuple, Type

import torch
from torch import Tensor, nn

from .common import MLPBlock


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input point cloud
        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block

        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        pointcloud_embedding: Tensor,
        pointcloud_emorigin: Tensor,
        question_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          pointcloud_embedding (torch.Tensor): image to attend to. Should be shape
            B x N x C
          question_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.
        Returns:
          torch.Tensor: the processed question_embedding
          torch.Tensor: the processed pointcloud_embedding
        """
        bs, N, c = pointcloud_embedding.shape
        queries = question_embedding
        keys = pointcloud_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=question_embedding,
                key_pe=pointcloud_emorigin,
            )

        # Apply the final attention layer from the points to the image
        q = queries + question_embedding
        k = keys + pointcloud_emorigin
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer

        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys



class TwoWayAttentionBlockNew(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_queries: int = 1,  # 可学习queries的数量
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        use_learnable_queries: bool = True,  # 是否使用可学习queries
    ) -> None:
        """
        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          num_queries (int): number of learnable query embeddings
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          use_learnable_queries (bool): whether to use learnable queries
        """
        super().__init__()
        
        self.use_learnable_queries = use_learnable_queries
        self.num_queries = num_queries
        
        # 可学习的query embeddings
        if use_learnable_queries:
            self.learnable_queries = nn.Parameter(
                torch.randn(1, num_queries, embedding_dim)
            )
            self.learnable_query_pe = nn.Parameter(
                torch.randn(1, num_queries, embedding_dim)
            )
        
        # Token聚合模块 - 处理多个token embeddings
        self.token_aggregation = TokenAggregation(embedding_dim, num_heads)
        
        # Token到Query的cross-attention（新增）
        self.cross_attn_token_to_query = Attention(embedding_dim, num_heads)
        self.norm_token_to_query = nn.LayerNorm(embedding_dim)
        
        # Self attention - 现在作用于learnable queries
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        # Cross attention token to image
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # MLP block
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        
        # Cross attention image to token
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, 
        token_embeddings: Tensor,  # [B, N_tokens, C] - 多个token embeddings
        keys: Tensor,              # [B, H*W, C] - 点云/图像特征
        key_pe: Tensor             # [B, H*W, C] - 点云/图像位置编码
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            token_embeddings: [B, N_tokens, C] 多个token的embeddings
            keys: [B, N_features, C] 点云或图像特征
            key_pe: [B, N_features, C] 特征的位置编码
        Returns:
            queries: [B, N_queries, C] 更新后的queries
            keys: [B, N_features, C] 更新后的特征
        """
        B = token_embeddings.shape[0]
        
        # 1. 聚合多个token embeddings
        aggregated_tokens = self.token_aggregation(token_embeddings)  # [B, 1, C]
        
        # 2. 初始化queries
        if self.use_learnable_queries:
            # 使用可学习的queries
            queries = self.learnable_queries.expand(B, -1, -1)  # [B, N_queries, C]
            query_pe = self.learnable_query_pe.expand(B, -1, -1)
            
            # Token信息通过cross-attention注入到queries中
            q = queries + query_pe
            attn_out = self.cross_attn_token_to_query(
                q=q, 
                k=token_embeddings,  # 使用所有token作为key
                v=token_embeddings
            )
            queries = queries + attn_out
            queries = self.norm_token_to_query(queries)
        else:
            # 使用聚合后的token作为初始queries
            queries = aggregated_tokens
            query_pe = torch.zeros_like(queries)
        
        # 3. Self attention block - queries之间的交互
        if self.skip_first_layer_pe:
            attn_out = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
        queries = queries + attn_out
        queries = self.norm1(queries)
        
        # 4. Cross attention block - queries attending to image/point cloud
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        
        # 5. MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        
        # 6. Cross attention block - image attending to queries
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        
        return queries, keys


class TokenAggregation(nn.Module):
    """聚合多个token embeddings的模块"""
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.attention = Attention(embedding_dim, num_heads)
        self.norm = nn.LayerNorm(embedding_dim)
        
        # 可学习的聚合query
        self.aggregation_query = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )
    
    def forward(self, token_embeddings: Tensor) -> Tensor:
        """
        Args:
            token_embeddings: [B, N_tokens, C]
        Returns:
            aggregated: [B, 1, C]
        """
        B = token_embeddings.shape[0]
        query = self.aggregation_query.expand(B, -1, -1)
        
        # 使用attention进行聚合
        aggregated = self.attention(
            q=query,
            k=token_embeddings,
            v=token_embeddings
        )
        aggregated = self.norm(query + aggregated)
        
        return aggregated

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class FeatureInjector(nn.Module):
    def __init__(self, token_dim=3072, point_dim=32, internal_dim=128):
        super().__init__()
        # 1. 适配层：先不急着压到 32，而是压到一个中间维度 (如 128)
        self.token_proj = nn.Linear(token_dim, internal_dim)
        self.point_proj_up = nn.Linear(point_dim, internal_dim) # 点云升维
        
        # 2. Attention 在 128 维上做 (head_dim = 128/4 = 32, 正常多了)
        self.cross_attn = nn.MultiheadAttention(embed_dim=internal_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(internal_dim)
        
        # 3. 输出投影：降回 32 维
        self.point_proj_down = nn.Linear(internal_dim, point_dim)
        
        # 4. 【关键】零初始化 (Zero Init)
        # 让这一层在训练开始时输出全 0，完全不影响原本的模型
        nn.init.zeros_(self.point_proj_down.weight)
        nn.init.zeros_(self.point_proj_down.bias)

    def forward(self, point_emb, raw_tokens):
        # point_emb: [B, N, 32]
        # raw_tokens: [B, K, 3072]
        
        # 1. 映射到中间维度
        q = self.point_proj_up(point_emb)  # [B, N, 128]
        k = v = self.token_proj(raw_tokens) # [B, K, 128]
        
        # 2. Cross Attention + Residual + Norm
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        # Standard Transformer Block Structure
        features = self.norm(q + attn_out) 
        
        # 3. 映射回原始维度
        delta = self.point_proj_down(features) # [B, N, 32]
        
        # 4. 再次残差连接 (这一步最开始是 point + 0)
        return point_emb + delta
