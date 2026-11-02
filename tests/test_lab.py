import unittest, tempfile, pathlib
import numpy as np
import torch
from forecast_lab.data import make_energy,make_windows,TrainScaler,input_visibility,load_etth1,origins_for
from forecast_lab.models import seasonal_naive,RidgeDirect,ridge_features,MiniMaskedForecaster
from forecast_lab.metrics import mae,mase,seasonal_scale,paired_origin_bootstrap_delta

class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.ds=make_energy(0)
    def test_reproducible_data(self):
        self.assertTrue(np.array_equal(self.ds.values,make_energy(0).values))
        self.assertFalse(np.array_equal(self.ds.values,make_energy(1).values))
    def test_shape_and_channels(self): self.assertEqual(self.ds.values.shape,(1488,6))
    def test_train_window_cannot_cross_cutoff(self):
        _,_,orig=make_windows(self.ds,96,24,96,self.ds.train_end,2)
        self.assertTrue(np.all(orig+24 <= self.ds.train_end))
    def test_holdout_windows_nonoverlap(self):
        for start,stop in origins_for(self.ds,24).values():
            _,_,orig=make_windows(self.ds,96,24,start,stop,24)
            self.assertTrue(np.all(np.diff(orig)==24))
            self.assertTrue(np.all(orig+24<=stop))
    def test_future_target_hidden(self):
        x=np.random.randn(2,120,6).astype('float32')
        a,m=input_visibility(x,96)
        self.assertTrue(np.all(a[:,96:,:4]==0))
        self.assertTrue(np.all(m[:,96:,:4]==0))
        self.assertTrue(np.array_equal(a[:,96:,4:],x[:,96:,4:]))
    def test_future_target_invariance(self):
        x=np.random.randn(2,120,6).astype('float32')
        x2=x.copy();x2[:,96:,:4]+=1_000_000
        a,m=input_visibility(x,96);b,n=input_visibility(x2,96)
        self.assertTrue(np.array_equal(a,b))
    def test_train_only_scaling(self):
        ds=make_energy(0);s=TrainScaler().fit(ds)
        base=s.mu.copy();ds.values[1100:]+=10000
        self.assertTrue(np.array_equal(base,TrainScaler().fit(ds).mu))
    def test_naive_repeats_last_day(self):
        x,y,_=make_windows(self.ds,96,24,864,888,24)
        p=seasonal_naive(x,96,24)
        self.assertTrue(np.array_equal(p[0].T,x[0,72:96,:3]))
    def test_ridge_shape(self):
        x,y,_=make_windows(self.ds,96,24,96,320,8)
        f=ridge_features(x,96,24)
        model=RidgeDirect().fit(f,y)
        self.assertEqual(model.predict(f,3,24).shape,y.shape)
    def test_mase_zero(self):
        y=np.ones((3,3,24));self.assertEqual(mase(y,y,np.ones(3)),0.)
    def test_mae_one(self):self.assertEqual(mae(np.array([0,0]),np.array([1,-1])),1.)
    def test_scale_positive(self):self.assertTrue(np.all(seasonal_scale(self.ds.values[:864,:3])>0))
    def test_bootstrap_paired_zero(self):
        y=np.ones((5,3,24));result=paired_origin_bootstrap_delta(y,y,y)
        self.assertEqual(result['ci95'],[0.0,0.0])
    def test_model_output_shape(self):
        torch.manual_seed(0)
        m=MiniMaskedForecaster(d=16,heads=4,layers=1)
        x=torch.randn(2,120,6);mask=torch.ones_like(x)
        self.assertEqual(m(x,mask).shape,(2,3,24))
    def test_temporal_causality(self):
        torch.manual_seed(0)
        m=MiniMaskedForecaster(d=16,heads=4,layers=1).eval()
        x=torch.randn(1,120,6);mask=torch.ones_like(x)
        x2=x.clone();x2[:,112:,4:]+=100
        with torch.no_grad():
            a=m(x,mask);b=m(x2,mask)
        self.assertTrue(torch.allclose(a[:,:,:8],b[:,:,:8],atol=1e-5))
    def test_real_csv_parser(self):
        from datetime import datetime,timedelta
        with tempfile.TemporaryDirectory() as temp:
            p=pathlib.Path(temp)/'data.csv'
            p.write_text('date,HUFL,MUFL,OT,HULL\n'+''.join(
              f'{(datetime(2020,1,1)+timedelta(hours=i)):%Y-%m-%d %H:%M:%S},1,2,3,4\n'
               for i in range(600)))
            out=load_etth1(p)
            self.assertEqual(out.values.shape,(600,6))
            self.assertEqual(out.train_end,360)

if __name__=='__main__':unittest.main()
