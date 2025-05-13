import pandas as pd
import numpy as np

def _get_legacy_data_1460(self):
    """
    Function to read the legacy data of the 1060 grid computed by Roxana Lupu.

    Note
    ----
    This function is **highly** sensitive to the file format. You cannot edit the ascii file and then
    run this function. Each specific line is accounted for.
    """
    data = pd.read_csv(self.ck_file,
              sep='\s+',header=None,
              names=list(range(9)),dtype=str)

    num_species = int(data.iloc[0,0])
    if num_species == 24:
        max_ele = 35
        self.max_tc = 73
        self.max_pc = 20
        max_windows = 200

        self.molecules = [str(data.iloc[i,j]) for i in [0,1,2]
        for j in range(9)][1:num_species+1]

        last = [float(data.iloc[int(max_ele*self.max_pc*self.max_tc/3)+3,0])]
        end_abunds = 3+int(max_ele*self.max_pc*self.max_tc/3)

        abunds = list(np.array(
            data.iloc[3:end_abunds,0:3].astype(float)
            ).ravel())
        abunds = abunds + last
        abunds = np.reshape(abunds,(self.max_pc,self.max_tc,max_ele),order='F')

        self.nwno = int(data.iloc[end_abunds,1])

        end_window = int(max_windows/3)
        self.wno = (data.iloc[end_abunds:end_abunds+end_window,0:3].astype(float)).values.ravel()[2:]
        if float(data.iloc[end_abunds+end_window+1,2]) == 0.0: # for >+1.0 metallicity tables
            max_windows=1000
            offset_new=267
            self.delta_wno = (data.iloc[end_abunds+end_window+1+offset_new:1+end_abunds+2*end_window+offset_new,0:3].astype(float)).values.ravel()[:-2]
            end_windows =1+end_abunds+2*end_window+2*(offset_new)

            nc_t=int(data.iloc[end_windows,1])

            self.nc_p = (np.zeros(shape=(nc_t))+20).astype(int)

            end_npt = 1+end_windows+int(self.max_tc/6) + 11 #12 dummy rows

            #first = list(data.iloc[end_npt,4:5].astype(float))



            self.pressures = np.array(list(np.array(data.iloc[end_npt+1:end_npt + int(self.max_pc*self.max_tc/3)+2,0:3]
                                .astype(float))
                                .ravel()[:-1]))/1e3

            end_ps = end_npt + int(self.max_pc*self.max_tc/3)

            self.temps = list(np.array(data.iloc[end_ps+1:2+int(end_ps+nc_t/3),0:3]
                        .astype(float))
                        .ravel()[2:])

            end_temps = int(end_ps+nc_t/3)+2

            ngauss1, ngauss2,  =data.iloc[end_temps,0:2].astype(int)
            gfrac = float(data.iloc[end_temps,2])
            self.ngauss = int(data.iloc[end_temps,3])

            assert self.ngauss == 8, 'Legacy code uses 8 gauss points not {0}. Check read in statements'.format(self.ngauss)

            gpts_wts = np.reshape(np.array(data.iloc[end_temps+1:2+end_temps+int(2*self.ngauss/3),0:3]
             .astype(float)).ravel()[:-2], (self.ngauss,2))

            self.gauss_pts = np.array([i[0] for i in gpts_wts])
            self.gauss_wts = np.array([i[1] for i in gpts_wts])

            kappa = np.array(
                data.iloc[3+end_temps+int(2*self.ngauss/3):-2,0:3]
                    .astype(float)).ravel()[0:-1]

            kappa = np.reshape(kappa,
                        (max_windows,self.ngauss*2,self.max_pc,self.max_tc),order='F')

        #want the axes to be [npressure, ntemperature, nwave, ngauss ]
            kappa = kappa.swapaxes(1,3)
            kappa = kappa.swapaxes(0,2)
            self.kappa = kappa[:, :, 0:self.nwno, 0:self.ngauss]

        #finally add pressure/temperature scale to abundances
            self.full_abunds['pressure']= self.pressures[self.pressures>0]
            self.full_abunds['temperature'] = np.concatenate([[i]*max(self.nc_p) for i in self.temps])[self.pressures>0]


        else:

            self.delta_wno = (data.iloc[end_abunds+end_window+1:1+end_abunds+2*end_window,0:3].astype(float)).values.ravel()[1:-1]
            end_windows =2+end_abunds+2*end_window

            nc_t=int(data.iloc[end_windows,0])
        #this defines the number of pressure points per temperature grid
        #historically not all pressures are run for all temperatures
        #though in 1460 there are always 20
            self.nc_p = np.array(data.iloc[end_windows:1+end_windows+int(self.max_tc/6),0:6].astype(int
                    )).ravel()[1:-4]

            end_npt = 1+end_windows+int(self.max_tc/6) + 11 #11 dummy rows

            first = list(data.iloc[end_npt,4:5].astype(float))

        #convert to bars
            self.pressures = np.array(first+list(np.array(data.iloc[end_npt+1:end_npt + int(self.max_pc*self.max_tc/3) + 2,0:3]
                                .astype(float))
                                .ravel()[0:-2]))/1e3

            end_ps = end_npt + int(self.max_pc*self.max_tc/3)

            self.temps = list(np.array(data.iloc[end_ps+1:2+int(end_ps+nc_t/3),0:3]
                        .astype(float))
                        .ravel()[1:-1])
            end_temps = int(end_ps+nc_t/3)+1


            ngauss1, ngauss2,  =data.iloc[end_temps,2:4].astype(int)
            gfrac = float(data.iloc[end_temps+1,0])
            self.ngauss = int(data.iloc[end_temps+1,1])

            assert self.ngauss == 8, 'Legacy code uses 8 gauss points not {0}. Check read in statements'.format(self.ngauss)

            gpts_wts = np.reshape(np.array(data.iloc[end_temps+1:2+end_temps+int(2*self.ngauss/3),0:3]
               .astype(float)).ravel()[2:], (self.ngauss,2))

            self.gauss_pts = [i[0] for i in gpts_wts]
            self.gauss_wts = [i[1] for i in gpts_wts]

            kappa = np.array(
               data.iloc[3+end_temps+int(2*self.ngauss/3):-2,0:3]
                    .astype(float)).ravel()[0:-2]
            kappa = np.reshape(kappa,
                        (max_windows,self.ngauss*2,self.max_pc,self.max_tc),order='F')

        #want the axes to be [npressure, ntemperature, nwave, ngauss ]
            kappa = kappa.swapaxes(1,3)
            kappa = kappa.swapaxes(0,2)
            self.kappa = kappa[:, :, 0:self.nwno, 0:self.ngauss]

        #finally add pressure/temperature scale to abundances
            self.full_abunds['pressure']= self.pressures[self.pressures>0]
            self.full_abunds['temperature'] = np.concatenate([[i]*max(self.nc_p) for i in self.temps])[self.pressures>0]

    elif num_species == 22:
        print("NOTE: You are loading Opacity tables without any Gaseous TiO and VO opacities")
        max_ele = 35
        self.max_tc = 73
        self.max_pc = 20
        max_windows = 200

        self.molecules = [str(data.iloc[i,j]) for i in [0,1,2]
        for j in range(9)][1:num_species+1]

        last = [float(data.iloc[int(max_ele*self.max_pc*self.max_tc/3)+1,0])]
        end_abunds = 3+int(max_ele*self.max_pc*self.max_tc/3)


        abunds = list(np.array(
            data.iloc[3:end_abunds,0:3].astype(float)
            ).ravel())
        abunds = abunds + last
        abunds = np.reshape(abunds,(self.max_pc,self.max_tc,max_ele),order='F')


        self.nwno = int(data.iloc[end_abunds,0])


        end_window = int(max_windows/3)

        self.wno = (data.iloc[end_abunds:end_abunds+end_window,0:3].astype(float)).values.ravel()[1:-1]
        if float(data.iloc[end_abunds+end_window+1,2]) == 0.0: # for >+1.0 metallicity tables
            max_windows=1000
            offset_new=266
            self.delta_wno = (data.iloc[end_abunds+end_window+1+offset_new:1+end_abunds+2*end_window+offset_new,0:3].astype(float)).values.ravel()[2:]
            end_windows =1+end_abunds+2*end_window+2*(offset_new+1)

            nc_t=int(data.iloc[end_windows,0])

            self.nc_p = (np.zeros(shape=(nc_t))+20).astype(int)
            end_npt = 1+end_windows+int(self.max_tc/6) + 12-1 #12 dummy rows

            first = list(data.iloc[end_npt,4:5].astype(float))



            self.pressures = np.array(first+list(np.array(data.iloc[end_npt+1:end_npt + int(self.max_pc*self.max_tc/3)+2,0:3]
                                .astype(float))
                                .ravel()[:-2]))/1e3

            end_ps = end_npt + int(self.max_pc*self.max_tc/3)

            self.temps = list(np.array(data.iloc[end_ps+1:2+int(end_ps+nc_t/3),0:3]
                        .astype(float))
                        .ravel()[1:-1])
            end_temps = int(end_ps+nc_t/3)+1

            ngauss1, ngauss2,  =data.iloc[end_temps,2:4].astype(int)
            gfrac = float(data.iloc[end_temps+1,0])
            self.ngauss = int(data.iloc[end_temps+1,1])

            assert self.ngauss == 8, 'Legacy code uses 8 gauss points not {0}. Check read in statements'.format(self.ngauss)

            gpts_wts = np.reshape(np.array(data.iloc[end_temps+1:2+end_temps+int(2*self.ngauss/3),0:3]
             .astype(float)).ravel()[2:], (self.ngauss,2))

            self.gauss_pts = [i[0] for i in gpts_wts]
            self.gauss_wts = [i[1] for i in gpts_wts]

            kappa = np.array(
                data.iloc[3+end_temps+int(2*self.ngauss/3):-2,0:3]
                    .astype(float)).ravel()[0:-1]

            kappa = np.reshape(kappa,
                        (max_windows,self.ngauss*2,self.max_pc,self.max_tc),order='F')

        #want the axes to be [npressure, ntemperature, nwave, ngauss ]
            kappa = kappa.swapaxes(1,3)
            kappa = kappa.swapaxes(0,2)
            self.kappa = kappa[:, :, 0:self.nwno, 0:self.ngauss]

        #finally add pressure/temperature scale to abundances
            self.full_abunds['pressure']= self.pressures[self.pressures>0]
            self.full_abunds['temperature'] = np.concatenate([[i]*max(self.nc_p) for i in self.temps])[self.pressures>0]



        else:
            self.delta_wno = (data.iloc[end_abunds+end_window+1:1+end_abunds+2*end_window,0:3].astype(float)).values.ravel()[:-2]

            end_windows =1+end_abunds+2*end_window
            nc_t=int(data.iloc[end_windows,2])
        #this defines the number of pressure points per temperature grid
        #historically not all pressures are run for all temperatures
        #though in 1460 there are always 20

        #self.nc_p = np.array(data.iloc[end_windows:1+end_windows+int(self.max_tc/6),:]).ravel()[3:].astype(int)
        # noTiOVO format is such that it is not in 6x table
            self.nc_p = (np.zeros(shape=(nc_t))+20).astype(int)
            end_npt = 1+end_windows+int(self.max_tc/6) + 12 #12 dummy rows

            first = list(data.iloc[end_npt,2:4].astype(float))


        #convert to bars
            self.pressures = np.array(first+list(np.array(data.iloc[end_npt+1:end_npt + int(self.max_pc*self.max_tc/3)+1,0:3]
                                .astype(float))
                                .ravel()[0:]))/1e3


            end_ps = end_npt + int(self.max_pc*self.max_tc/3)

            self.temps = list(np.array(data.iloc[end_ps+1:2+int(end_ps+nc_t/3),0:3]
                        .astype(float))
                        .ravel()[:-2])
            end_temps = int(end_ps+nc_t/3)+1


            ngauss1, ngauss2,  =data.iloc[end_temps,1:3].astype(int)
            gfrac = float(data.iloc[end_temps,3])
            self.ngauss = int(data.iloc[end_temps+1,0])


            assert self.ngauss == 8, 'Legacy code uses 8 gauss points not {0}. Check read in statements'.format(self.ngauss)

            gpts_wts = np.reshape(np.array(data.iloc[end_temps+1:2+end_temps+int(2*self.ngauss/3),0:3]
             .astype(float)).ravel()[1:-1], (self.ngauss,2))

            self.gauss_pts = np.array([i[0] for i in gpts_wts])
            self.gauss_wts = np.array([i[1] for i in gpts_wts])

            kappa = np.array(
                data.iloc[3+end_temps+int(2*self.ngauss/3):-2,0:3]
                    .astype(float)).ravel()[0:-2]

            kappa = np.reshape(kappa,
                        (max_windows,self.ngauss*2,self.max_pc,self.max_tc),order='F')

        #want the axes to be [npressure, ntemperature, nwave, ngauss ]
            kappa = kappa.swapaxes(1,3)
            kappa = kappa.swapaxes(0,2)
            self.kappa = kappa[:, :, 0:self.nwno, 0:self.ngauss]

        #finally add pressure/temperature scale to abundances
            self.full_abunds['pressure']= self.pressures[self.pressures>0]
            self.full_abunds['temperature'] = np.concatenate([[i]*max(self.nc_p) for i in self.temps])[self.pressures>0]
