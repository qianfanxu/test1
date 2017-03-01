package SMOTest2.svm;

import SMOTest2.svm.kernel.GaussianKernel;
import SMOTest2.svm.kernel.Kernel;
import SMOTest2.svm.kernel.LinearKernel;
import SMOTest2.svm.vector.DataVector;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * Created by xuqf on 2017/2/23.
 */
public class SVDD {

    private static final long serialVersionUID = 7667970753574953210L;
    public static final double EPSILON = 1e-3;

    List<SupportVector> vectors = new ArrayList<SupportVector>();

    Kernel kernel;
    //double R = 0;       // raduis
    double R_Square = 0;
    //double a = 0;       // center of the ball
    double a_Square = 0;       // center of the ball
    double c;           // penalty factor

    // For init iteration, alpha_i = 1/n, a = 0, R=0


    public double getR() {
        return Math.sqrt(R_Square);
    }

    public double getR_Square() {
        return R_Square;
    }

    public double getA() {
        return Math.sqrt(a_Square);
    }

    public double getA_Square() {
        return a_Square;
    }

    public SVDD(Kernel kernel, double c) {
        this.kernel = kernel;
        this.c = c;
    }

    public SVDD(Kernel kernel) {
        this(kernel, Double.POSITIVE_INFINITY);
    }


    public SVDD() {
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


    /*
    *
    *   EG: i = 1
    *    d_1^2 = ||pha(x_1) - a||^2
    *  = K_11 -2a_1*K11 - 2a_2*K21 - 2q_1( q_1 = a3*K13 + a4*K14 + a5*K15 + ......) + a^2
    *  =
    * */
    public double output(SupportVector v1){

        double sum = kernel.getValue(v1.x, v1.x) + a_Square;
        for(SupportVector v: vectors){

            if( v.alpha <= SVM.EPSILON){
                continue;
            }
            sum -= 2*(v.alpha)*(kernel.getValue(v.x, v1.x));

        }
        //System.out.println("debug: output sum = " + sum);
        return sum;
    }

    /*
    *   d_i^2 = ||pha(x_i) - a||^2
    *   =  phi(x_i)^2 - 2a
    *   = K_ii - 2*Sigma(1,n)(K_ij) + a^2
    *   = K_ii + a^2 - 2*Sigma(1,n)(K_ij)
    *
    *   EG: i = 1
    *    d_1^2 = ||pha(x_1) - a||^2
    *  = K_11 + K
    * */
    public double predict(DataVector x){

        double sum = kernel.getValue(x,x) + a_Square;
        for(SupportVector v: vectors){

            if( v.alpha <= SVM.EPSILON){
                continue;
            }
            sum -= 2*(v.alpha)*(kernel.getValue(v.x, x));

        }
        //System.out.println("debug: output sum = " + sum);
        return (sum - R_Square);
    }

    public static void main(String[] args) {

        // create a new Gaussian SVM with C=100
        SVDD svdd = new SVDD(new GaussianKernel(), 100);
        Random random = new Random();

        // add 50 noisy examples of each input-output pair
        // double[][] xs = {{-1,-1},{+1,-1},{-1,+1},{+1,+1}};
        double[][] trainData = {{0.9414832749366958,0.016894428796575958},{0.05089264084610037,0.7473105453820907},
                {0.1599332769719465,-0.12902689940067658},{-0.9520110117699603,0.06467902405954089},{0.8140863860003457,-0.20353463144583783},
                {0.4745859295605214,0.27568398616232503},{0.5882332416354952,-0.7643390058380277},{-0.41219259897398464,-0.4788621411764126},
                {0.3453347684787988,0.866959887125576},{-0.28198948658834944,-0.6883905090428193},{-0.2809063347819961,-0.9068078740775043},
                {0.29831315689332216,0.5884317045268844},{-0.06792447202933331,-0.8683690331877634},{0.33454025033630286,0.3294559600134583},
                {0.09660312653912584,0.07761891868960183},{0.2518241423927387,-0.45303918851295233},{0.4513591826304036,-0.04453338506105966},
                {0.30298796502797043,0.6561604694993846},{0.9248969209900174,0.5797396086009687},{0.037748054773162905,0.06699936686437728},
                {0.42407050560526577,-0.28723618111754745},{-0.5276438093071573,0.19419343422553975},{0.6316659529155485,0.4837429380480069},
                {-0.8926667017766166,0.26972541965896474},{0.7162077999458943,0.04631534496508811},{0.9157307311891555,-0.11045257489061139},
                {-0.1789095442191683,-0.9891041378063342},{0.1717174917559537,-0.07068907893280857},{0.6963530676845728,0.7296641891481583},
                {-0.31044321032623384,-0.030724702182547445},{0.38558742198736673,-0.2426726574250217},{0.9688844797024006,-0.04858840903622606},
                {-0.7698450483033924,-0.13234940393181724},{0.4625838269834729,0.628071470256639},{-0.7651351357480806,0.1674756639125914},
                {-0.07756257091627813,-0.8703733513359332},{-0.0864565549059403,0.5439895553191467},{0.9426462392551044,-0.3287702960668551},
                {-0.3215592803539693,0.5031699069820773},{-0.9427726594995014,0.5263496627241757},{0.2966190528972262,0.18522423641383726},
                {-0.9099413180530118,0.4346849925366573},{-0.2645787072623493,0.19076990148284703},{-0.3011035345349396,0.45526135771400095},
                {0.16208779804205822,-0.40699649607220545},{-0.8118179905258611,-0.6581422925022067},{-0.12442848254309569,-0.020524385684352244},
                {-0.012377446410381845,0.108077684755052},{0.10290634140126868,0.7856973452486274},{0.17014391179831956,-0.16334946766505468}};


        for(int j = 0; j < 50; j++) {
            // add some input
            double x1 = trainData[j][0];
            double x2 = trainData[j][1];
            DataVector v = new DataVector( x1, x2 );

            svdd.add(v, 1);
        }

        SVDDSMO.train(svdd);

        double[][] xs = {{-1,-1.2},{+0.9,-0.9},{+0.5,-0.9},{-1,+1.1},{+1,+2},{+0.1, +0.1},{+0.2, +0.2},{-0.2, +0.2},{+0.2, -0.2}};
        for(double[] x : xs) {
            // convert array to vector
            DataVector v = new DataVector(x);
            // calculate output of SVM
            double u = svdd.predict(v);
            System.out.println(v + " : " + u);
        }


        System.out.println( " a " + svdd.getA());
        System.out.println( " A_Square " + svdd.getA_Square());
        System.out.println( " R " + svdd.getR());
        System.out.println( " R_Square " + svdd.getR_Square());

       /* System.out.println("--------------------------------------");

        int index = 0;
        for(SupportVector v: svdd.vectors){
            if( !v.bound ) {
                if(v.alpha!=0){
                    System.out.println( index + " alpha " + v.alpha + "      "+v.x);
                }

            }
            index++;
        }

        System.out.println("--------------------------------------");

        index = 0;
        for(SupportVector v: svdd.vectors){
            if( !v.bound ) {
                if(v.alpha==0){
                    System.out.println( index + " alpha " + v.alpha + "      "+v.x);
                }

            }
            index++;
        }
        */

    }

//  Random Create double[][] for static test
//
//      System.out.print("double[][] xs = {");
//        for(int j = 0; j < 50; j++) {
//        // add some input
//        double x1 = random.nextGaussian()/2;
//        double x2 = random.nextGaussian()/2;
//        DataVector v = new DataVector( x1, x2 );
//
//        System.out.print("{" + x1 + "," + x2 + "},");
//
//        svdd.add(v, 1);
//        }
//        System.out.print("};");
    



}
