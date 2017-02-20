/*
* Copyright (C) 2010-2011 David A Roberts <d@vidr.cc>
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

package SMOTest2.svm;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import SMOTest2.svm.kernel.GaussianKernel;
import SMOTest2.svm.kernel.Kernel;
import SMOTest2.svm.kernel.LinearKernel;
import SMOTest2.svm.vector.DataVector;

/**
 * Created by xuqf on 2017/2/15.
 */
public class SVM implements Serializable {
    private static final long serialVersionUID = 7667970753574953210L;
    public static final double EPSILON = 1e-3;

    List<SupportVector> vectors = new ArrayList<SupportVector>();

    double b = 0;

    Kernel kernel;

    double c;
    

    public SVM(Kernel kernel, double c) {
        this.kernel = kernel;
        this.c = c;
    }
    

    public SVM(Kernel kernel) {
        this(kernel, Double.POSITIVE_INFINITY);
    }
    

    public SVM() {
        this(new LinearKernel());
    }
    

    public void add(DataVector x, int y) {
        vectors.add(new SupportVector(x, y));
    }
    

    public void prune() {
        Iterator<SupportVector> iter = vectors.iterator();
        while(iter.hasNext())
            if(iter.next().alpha <= EPSILON)
                iter.remove();
    }
    

    public int size() {
        return vectors.size();
    }

    /*
    public double output(DataVector x) {
        double u = -b;
        for(SupportVector v : vectors) {
            if(v.alpha <= EPSILON)
                continue;
                u += v.alpha * v.y * kernel.getValue(v.x, x);
        }
        return u;
    }
    */

    public double output(SupportVector v1){

        // $u = sigmaOf(i)*ai*yi*K(xi, x) - b$
        double sum = -b;
        for(SupportVector v: vectors){

            if( v.alpha <= SVM.EPSILON){
                continue;
            }
            sum += (v.alpha)*(v.y)*(kernel.getValue(v.x, v1.x));

        }
        //System.out.println("debug: output sum = " + sum);
        return sum;
    }

    public double predict(DataVector x){

        // $u = sigmaOf(i)*ai*yi*K(xi, x) - b$
        double sum = -b;
        for(SupportVector v: vectors){

            if( v.alpha <= SVM.EPSILON){
                continue;
            }
            sum += (v.alpha)*(v.y)*(kernel.getValue(v.x, x));
        }
        return sum;
    }


    public static void main(String[] args) {

        double[][] xs = {{-1,-1},{+1,-1},{-1,+1},{+1,+1}};
        int[]      ys = {    -1,     +1,     +1,     -1 };
        
        // create a new Gaussian SVM with C=100
        SVM svm = new SVM(new GaussianKernel(), 100);
        Random random = new Random();
        
        // add 50 noisy examples of each input-output pair
        for(int i = 0; i < 4; i++) {
            double[] x = xs[i];
            int y = ys[i];
            for(int j = 0; j < 50; j++) {
                // add some input
                DataVector v = new DataVector(
                        x[0] + random.nextGaussian()/2,
                        x[1] + random.nextGaussian()/2);
                svm.add(v, y);
            }
        }
        

        SMO.train(svm);

        for(double[] x : xs) {
            // convert array to vector
            DataVector v = new DataVector(x);
            // calculate output of SVM
            double u = svm.predict(v);
            System.out.println(v + " : " + u);
        }
    }
}
