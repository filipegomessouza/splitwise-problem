from typing import Optional
import numpy as np

class RandomKeys:
    """One key per person, and the order they impose on a set of people.

    Shared by every constructor that wants a random-key encoding, so that two of them can
    be handed the very same chromosome and differ in nothing but what they build from it.

    The chromosome is optional, because it only makes sense once the instance is known --
    it needs exactly one key per person. Without one, a fresh set is drawn for whatever
    instance turns up, which is what lets a single object serve a whole benchmark.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._keys: Optional[np.ndarray] = None

    @property
    def keys(self) -> Optional[np.ndarray]:
        return self._keys

    def set(self, keys: Optional[np.ndarray]) -> None:
        if keys is None:
            self._keys = None

            return

        chromosome = np.asarray(keys, dtype=np.float64)

        if chromosome.ndim != 1:
            raise ValueError('random_keys must be one-dimensional')

        if chromosome.size and (chromosome.min() < 0.0 or chromosome.max() > 1.0):
            raise ValueError('random_keys must lie between 0 and 1')

        # no length check on purpose: swapping in a chromosome of another length is how
        # one object serves instances of different sizes
        self._keys = chromosome

    def fits(self, people: int) -> bool:
        """Whether this chromosome can decode an instance of that many people."""
        # without a chromosome any instance is fair game, since one gets drawn to fit
        return self._keys is None or len(self._keys) == people

    def order(self, people: int, subset: np.ndarray) -> np.ndarray:
        """The people of `subset`, sorted by their keys.

        `subset` is carried through rather than replaced, so the order comes back in the
        caller's index space and needs no remapping. Stable, so that equal keys keep person
        order and a run stays reproducible.
        """
        keys = self._keys_for(people)

        return subset[np.argsort(keys[subset], kind='stable')]

    def _keys_for(self, people: int) -> np.ndarray:
        """The chromosome to use, drawn on the spot if there is none.

        A drawn chromosome stays local to the call and is never kept: holding on to it
        would leave the object sized to whichever instance came first, and every instance
        of another size would stop being supported from then on.
        """
        if self._keys is None:
            return self._rng.random(people)

        if len(self._keys) != people:
            raise ValueError(
                f"{len(self._keys)} random keys cannot decode an instance of "
                f"{people} people"
            )

        return self._keys
