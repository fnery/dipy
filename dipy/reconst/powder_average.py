import numpy as np
from dipy.core.gradients import unique_bvals, btensor_to_bdelta

class PowderAverage():
    """Class for powder averaging diffusion MRI data
    """

    def __init__(self, gtab, bmag=None):
        """Initialise an instance of the PowderAverage class

        Parameters
        ----------
        gtab : a GradientTable class instance
            The gradient table containing diffusion acquisition parameters.
        bmag : int
            The order of magnitude that the bvalues have to differ to be
            considered an unique b-value. B-values are also rounded up to this
            order of of magnitude. Default: derive this value from the maximal
            b-value provided: $bmag=log_{10}(max(bvals)) - 1$.

        Attributes
        ----------
        idxs : list
            indices of measurements at each (bval, bdelta) combination
        n_meas : list
            number of measurements at each (bval, bdelta) combination
        bdeltas : list
            bdelta (i.e. b-tensor anisotropy) at each (bval, bdelta) combination
        bvals : list
            bval (i.e. b-value) at each (bval, bdelta) combination

        See Also
        --------
        gradient_table

        Notes
        --------
        This works both for linear tensor encoding (i.e. conventional) diffusion
        MRI data or data with multiple encoding types (linear/planar/spherical
        tensor encoding)

        """
        # Get bdeltas for each measurement
        if gtab.btens is None:
            # If the `btens` attribute from GradientTable was not specified,
            # let's assume that the data was acquired with conventional
            # diffusion encoding (i.e. linear tensor encoding). This way, we can
            # avoid more complex loops and conditionals below
            bdeltas = np.ones(len(gtab.bvals))
        else:
            bdeltas = btensor_to_bdelta(gtab.btens)[0]

        # Get unique bdeltas, assuming btensor_to_bdelta returns rounded bdeltas
        u_bdeltas = np.unique(bdeltas)

        # Initialise output lists
        idxs = []
        n_meas = []
        bdeltas_pa = []
        bvals_pa = []

        # Find measurement indices for each (bdelta, bval) combination
        for bdelta in u_bdeltas:
            i_bdelta_idxs = np.where(bdeltas == bdelta)[0]
            i_bvals = gtab.bvals[i_bdelta_idxs]
            iu_bvals, ir_bvals = unique_bvals(i_bvals, bmag=bmag, rbvals=True)

            for bval in iu_bvals:
                bval_idxs = i_bdelta_idxs[np.where(ir_bvals == bval)[0]]
                idxs.append(bval_idxs)
                n_meas.append(len(bval_idxs))
                bdeltas_pa.append(bdelta)
                bvals_pa.append(bval)

        self.gtab = gtab
        self.bmag = bmag
        self.idxs = idxs
        self.n_meas = n_meas
        self.bdeltas = bdeltas_pa
        self.bvals = bvals_pa


    def calculate(self, data):
        """Calculate powder average (pa) of diffusion MRI data

        Parameters
        ----------
        data : numpy.ndarray
            Measured signal from a single voxel, 1D array of shape (N,)
            OR
            Measured signal of all voxels, 4D array of shape (:,:,:,N)
            In both cases N = number of measurements in the dataset

        Returns
        -------
        array
            Powder-averaged signal from a single voxel, 1D array of shape (N_pa,)
            OR
            Powder-averaged signal of all voxels, 4D array of shape (:,:,:,N_pa)
            In both cases N_pa = sum of the number of unique bvalues at each
            unique bdelta

        Examples
        --------
        >>> pa = PowderAverage(gtab, bmag=BMAG)
        >>> data_pa = pa.calculate(data)
        # TODO: Finish examples

        """
        # Ensure `data` has valid number of dimensions
        ndim = np.ndim(data)
        if ndim not in (1, 4):
            raise ValueError("`data` should have 1 or 4 dimensions")

        # Pre-allocate data_pa
        if ndim == 1:
            data_pa = np.empty(len(self.bvals))
        else:
            data_pa = np.empty(data.shape[:3] + (len(self.bvals),))

        # Compute powder average for each bval, bdelta combination
        for i, i_idxs in enumerate(self.idxs):
            if ndim == 1:
                data_pa[i] = np.mean(data[i_idxs])
            else:
                data_pa[:, :, :, i] = np.mean(data[:, :, :, i_idxs], axis=3)

        return data_pa