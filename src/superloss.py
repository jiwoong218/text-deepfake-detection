import numpy
import torch
from scipy.special import lambertw
from typing import Any, Tuple, Union


class SuperLoss(torch.nn.Module):
	# L = (loss - tau) * sigma + lambda * (log(sigma) ** 2)
	def __init__(
		self, lam: float = 1.0, tau: float = 0.5, mom: float = 0.1, mu: float = 0.0
	) -> None:
		super(SuperLoss, self).__init__()
		assert 0 < lam and 0 <= mom <= 1 and 0 <= mu

		self.lam = float(lam)  # regularization hparam;	e.g., 1, 0.25
		self.tau = float(
			tau
		)  # threshold;	running average of input loss;	e.g., log(C) in classification
		self.mom = float(mom)  # e.g., 0.1
		self.mu = float(mu)

	def compute_sigma(self, loss: torch.Tensor) -> torch.Tensor:
		loss_numpy = loss.detach().cpu().numpy()
		eps = numpy.finfo(loss_numpy.dtype).eps

		if self.mom > 0:
			self.tau = (1.0 - self.mom) * self.tau + self.mom * loss_numpy.mean()

		beta = (loss_numpy - self.tau) / self.lam
		z = numpy.maximum(-numpy.exp(-1) + eps, beta * 0.5)

		sig = torch.from_numpy(numpy.exp(-lambertw(z).real)).to(
			dtype=loss.dtype, device=loss.device
		)

		return sig

	def forward(self, loss: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		sigma = self.compute_sigma(loss)
		superloss = sigma * loss
		return superloss, sigma


class HardFirstSuperLoss(SuperLoss):
	# L = (loss - tau) * sigma - lambda * (log(sigma) ** 2)

	def compute_sigma(self, loss: torch.Tensor) -> torch.Tensor:
		loss_numpy = loss.detach().cpu().numpy()
		eps = numpy.finfo(loss_numpy.dtype).eps

		if self.mom > 0:
			self.tau = (1.0 - self.mom) * self.tau + self.mom * loss_numpy.mean()

		beta = (loss_numpy - self.tau) / self.lam
		z = numpy.minimum(numpy.exp(-1) - eps, beta * 0.5)

		sig = torch.from_numpy(numpy.exp(-lambertw(-z).real)).to(
			dtype=loss.dtype, device=loss.device
		)

		return sig


class MediumFirstSuperLoss(SuperLoss):
	# L = (loss - tau) * sigma + sign(loss - tau) * lambda * ((log(sigma) ** 2) + 2 * sigma * (mu - exp(-1)))
	@staticmethod
	def _compute_sigma_internal(
		beta: numpy.ndarray, mu: float, mode: str
	) -> numpy.ndarray:
		if mode == "easy":
			z = 0.5 * beta + numpy.exp(-1) - mu
			sig = numpy.exp(-lambertw(-z).real)
		else:
			assert mode == "hard"
			z = 0.5 * beta - numpy.exp(-1) + mu
			sig = numpy.exp(-lambertw(z).real)

		return sig

	def compute_sigma(self, loss: torch.Tensor) -> torch.Tensor:
		loss_numpy = loss.detach().cpu().numpy()

		if self.mom > 0:
			self.tau = (1.0 - self.mom) * self.tau + self.mom * loss_numpy.mean()

		beta = (loss_numpy - self.tau) / self.lam
		sig = numpy.empty_like(beta)

		idx_easy = beta < 0
		idx_hard = ~idx_easy
		sig[idx_easy] = self._compute_sigma_internal(beta[idx_easy], self.mu, "easy")
		sig[idx_hard] = self._compute_sigma_internal(beta[idx_hard], self.mu, "hard")

		sig = torch.from_numpy(sig).to(dtype=loss.dtype, device=loss.device)

		return sig


class TwoEndsFirstSuperLoss(MediumFirstSuperLoss):
	# L = (loss - tau) * sigma - sign(loss - tau) * lambda * ((log(sigma) ** 2) + 2 * sigma * mu)

	@staticmethod
	def _compute_sigma_internal(
		beta: numpy.ndarray, mu: float, mode: str
	) -> numpy.ndarray:
		eps = numpy.finfo(beta.dtype).eps

		if mode == "easy":
			z = numpy.maximum(-numpy.exp(-1) + eps, beta * 0.5 + mu)
			sig = numpy.exp(-lambertw(z).real)
		else:
			assert mode == "hard"
			z = numpy.minimum(numpy.exp(-1) - eps, beta * 0.5 - mu)
			sig = numpy.exp(-lambertw(-z).real)

		return sig


def train_rnn_superloss(
	network: Union[torch.nn.Module, torch.nn.DataParallel],
	loader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	superloss: torch.nn.Module,
	epoch: int,
	device: torch.device,
	**kwargs: Any,
) -> None:
	network.train()

	for batch_idx, (x, y) in enumerate(loader):
		assert x.ndim == 3 and y.ndim == 2
		x = x.to(device)  # (B, T, D1)
		y = y.to(device)  # (B, 1)

		optimizer.zero_grad()
		pred = network(x)
		loss = torch.nn.functional.mse_loss(pred, y, reduction="none")

		sl, w = superloss(loss)
		sl.mean().backward()

		optimizer.step()

		print(
			f"\rEpoch {epoch:3d} {numpy.float32(batch_idx+1) / numpy.float32(len(loader)) * 100:3.2f} loss {loss.mean().tolist():.4f} sigma {w.mean().tolist():.4f} tau {superloss.tau:.4f}",
			end="",
		)
