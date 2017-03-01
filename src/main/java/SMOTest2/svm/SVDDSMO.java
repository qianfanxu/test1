package SMOTest2.svm;

import SMOTest2.util.MathUtil;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Created by xuqf on 2017/2/23.
 */
public class SVDDSMO {

    // random select the Vector to train
    private static final Random random = new Random();

    // svm to be trained
    private SVDD svdd;

    //error cache
    private Map<SupportVector, Double> errorCache = new HashMap<SupportVector, Double>();  //[???] 为什么HashMap 要使用包装类

    private SVDDSMO(SVDD svdd){
        this.svdd = svdd;
    }

    public static void train(SVDD svdd){

        SVDDSMO svddsmo = new SVDDSMO(svdd);
        svddsmo.train();
    }

    private void initSVDDParams(){
        this.svdd.R_Square = 0;
        this.svdd.a_Square = 0;
        double numVectors = this.svdd.size();
        for(SupportVector v: svdd.vectors){
            v.alpha = 1/numVectors;
        }
        for(SupportVector v: svdd.vectors){
            errorCache.put(v,svdd.output(v));
        }

    }


    private void train(){

        int numChanged = 0;
        boolean examineAll = true;

        initSVDDParams();

        // All data set and non-bound data set is alternatively progress
        while(examineAll || numChanged > 0){  // search for all and no v is changed.

            numChanged = 0;
            for(SupportVector v: svdd.vectors){

                if((examineAll || !v.bound) && examineExample(v)){
                    numChanged++;
                }
            }
            if(examineAll){
                examineAll = false;  // switch to non-bound
            }
            else if(numChanged == 0){
                examineAll = true;
            }

            System.out.println("-------------------------------");

            System.out.println(" numChanged =  " + numChanged);
            System.out.println(" examineAll =  " + examineAll);


            System.out.println("-------------------------------");

        }

    }


    private boolean examineExample(SupportVector v){

        if(satisfiyKKTCondition(v)){
            return false;               //没有违反KKT条件，需要选择下一个v
        }

        //选定一个步长比较大的 v （alpha2）
        if(!errorCache.isEmpty() && takeStep(secondChoice(error(v)), v))
            return true;

        //随机开始进行步长训练
        final int n = svdd.vectors.size();
        final int pos = random.nextInt(n);
        for( SupportVector v1: svdd.vectors.subList(pos, n) ){
            if(!v1.bound && takeStep(v1,v)){
                return true;
            }
        }

        for( SupportVector v1: svdd.vectors.subList(0, pos) ){
            if(!v1.bound && takeStep(v1,v)){
                return true;
            }
        }

        for( SupportVector v1: svdd.vectors.subList(0, pos) ){
            if(v1.bound && takeStep(v1,v)){
                return true;
            }
        }


        for( SupportVector v1: svdd.vectors.subList(0, pos) ){
            if(v1.bound && takeStep(v1,v)){
                return true;
            }
        }

        return false;
    }


    //找到步长最大的E2：即 |E1-E2|
    private SupportVector secondChoice(double error){

        Map.Entry<SupportVector,Double> best = null;

        if(error > 0){
            for(Map.Entry<SupportVector,Double> entry: errorCache.entrySet()){
                if(best == null || entry.getValue() < best.getValue() )
                    best = entry;
            }
        }
        else{

            for(Map.Entry<SupportVector,Double> entry: errorCache.entrySet()){
                if(best == null || entry.getValue() > best.getValue() )
                    best = entry;
            }
        }
        return best.getKey();
    }


    private boolean takeStep(SupportVector v1, SupportVector v2){

        if(v1.x == v2.x){
            return false;
        }

        final double alpha1_old = v1.alpha, alpha2_old = v2.alpha;
        //final double y1 = v1.y, y2 = v2.y;

        double l, h;

        // 确定 L H 上下界 [???] 为什么取这个最大最小值
        l = Math.max(0, alpha2_old + alpha1_old - svdd.c);
        h = Math.min(alpha2_old + alpha1_old, svdd.c);

        if(l == h)
            return false;

        // 对目标函数求导得出，对a2的一个更新
        final double k11 = svdd.kernel.getValue(v1.x,v1.x);
        final double k12 = svdd.kernel.getValue(v1.x,v2.x);
        final double k22 = svdd.kernel.getValue(v2.x,v2.x);

        // 对a2进行剪辑
        final double e1 = error(v1), e2 = error(v2);

        final double eta = k11 + k22 - 2*k12;

        if( eta < 0 ){
            throw new RuntimeException("violate Mercer's condition");
        }

        if( eta == 0 ){
            System.out.println("debug: eta == 0 ");
            return false;
        }

        v2.alpha = alpha2_old + (e1 - e2) / eta;
        //进行剪辑
        v2.alpha = MathUtil.window(v2.alpha, l, h);


        if(MathUtil.equals(v2.alpha, alpha2_old, SVDD.EPSILON*(v2.alpha+alpha2_old+SVDD.EPSILON))) {
            // change in alpha2 was too small
            v2.alpha = alpha2_old;
            return false;
        }

        v1.alpha = alpha1_old + (alpha2_old-v2.alpha);
        v1.bound = MathUtil.lessThan(v1.alpha, 0, SVDD.EPSILON) || MathUtil.greatThan(v1.alpha, svdd.c, SVDD.EPSILON);
        v2.bound = MathUtil.lessThan(v2.alpha, 0, SVDD.EPSILON) || MathUtil.greatThan(v2.alpha, svdd.c, SVDD.EPSILON);


        double a_Square_old = svdd.a_Square;
        //Tag[456]
        //更新阈值 a_Square  算法复杂度n*n
        svdd.a_Square = 0;
        for(SupportVector v_out: svdd.vectors){
            //if( ! v_out.bound ) {
                for(SupportVector v_inner: svdd.vectors){
                    //if( !v_out.bound ) {
                        svdd.a_Square += v_out.alpha * v_inner.alpha * svdd.kernel.getValue(v_out.x, v_inner.x);
                    //}
                }
            //}
        }

//[Tag][123]
//        double delta_aSquare = 0.0;
//        double delta_1 = 0.0;
//        double delta_1_Sum = 0.0;
//        double delta_2 = 0.0;
//        double delta_2_Sum = 0.0;
//        double a1_a1_K11 = 0.0;
//        double a2_a2_K22 = 0.0;
//        double a1_a2_K12 = 0.0;
//        double a2_a1_K21 = 0.0;
//
//        for(SupportVector v: svdd.vectors){
//
//            if( v == v1 ){
//                delta_1 = (v1.alpha - alpha1_old)*(v1.alpha - alpha1_old)*svdd.kernel.getValue(v1.x, v1.x);   // a_1*a_1*K_11
//                delta_2 = (v2.alpha - alpha2_old)*(v1.alpha - alpha1_old)*svdd.kernel.getValue(v2.x, v1.x);   // a_2*a_1*K_21
//                a1_a1_K11 = delta_1;
//                a2_a1_K21 = delta_2;
//            }
//            else if( v == v2 ){
//                delta_1 = (v1.alpha - alpha1_old)*(v2.alpha - alpha2_old)*svdd.kernel.getValue(v1.x, v1.x);   // a_1*a_2*K_12
//                delta_2 = (v2.alpha - alpha2_old)*(v2.alpha - alpha2_old)*svdd.kernel.getValue(v2.x, v1.x);   // a_2*a_2*K_22
//                a1_a2_K12 = delta_1;
//                a2_a2_K22 = delta_2;
//            }
//            else{
//                delta_1 = (v1.alpha - alpha1_old)*v.alpha*svdd.kernel.getValue(v1.x, v.x);   // a_1*a_2*K_11
//                delta_2 = (v2.alpha - alpha2_old)*v.alpha*svdd.kernel.getValue(v2.x, v.x);   // a_2*a_2*K_11
//            }
//            delta_1_Sum += delta_1;
//            delta_2_Sum += delta_2;
//        }
//
//        delta_aSquare = delta_1_Sum*2 + delta_2_Sum*2 - a1_a1_K11 - a2_a2_K22 - a1_a2_K12 - a2_a1_K21;
//        svdd.a_Square = a_Square_old + delta_aSquare;


        // 更新errorCache  d_1^2, d_2^2
        for(SupportVector v: svdd.vectors){

            if( v.bound ) continue;
            //if( v == v1 || v == v2 ) continue;
            final double k1 = svdd.kernel.getValue(v1.x, v.x);
            final double k2 = svdd.kernel.getValue(v2.x, v.x);
            double error = errorCache.get(v);
            //error += -(v1.alpha - alpha1_old)*k1 -(v2.alpha - alpha2_old)*k2 + delta_aSquare; // [Tag][123] svdd.a_Square - a_Square_old
            error += -(v1.alpha - alpha1_old)*k1 -(v2.alpha - alpha2_old)*k2 +  svdd.a_Square - a_Square_old;  //Tag[456]
            errorCache.put(v,error);
        }

        //更新R^2
        svdd.R_Square = Math.abs( errorCache.get(v1) + errorCache.get(v2) ) * 0.5;
        //svdd.R_Square = (Math.abs( errorCache.get(v1)) + Math.abs( errorCache.get(v2) ) )* 0.5;

        /*
        *  1. 只是储存无界（non-bound  0<ai<C）的向量
           2. 将所有做优化的 ai （joint optimization的 ai），error值全部置为0，否则移除
        *
        */

        if(!v1.bound) {
            //errorCache.put(v1,0.0);
            //errorCache.put(v1,svdd.R_Square);
        }
        else{
            errorCache.remove(v1);
        }

        if(!v2.bound) {
            //errorCache.put(v2,0.0);
            //errorCache.put(v2,svdd.R_Square);
        }
        else{
            errorCache.remove(v2);
        }
        System.out.println("------------------------------------------------------------");
        System.out.println( v1.x + "  alpha1_old = " + alpha1_old +"  ==>  v1.alpha = " + v1.alpha);
        System.out.println( v2.x + "  alpha2_old = " + alpha2_old +"  ==>  v2.alpha = " + v2.alpha);

        return true;
    }


    /*
    *
    * 把违背KKT条件的ai作为第一个
     * 满足KKT条件的情况是：
     * 【1】. d_i^2 <= R^2 and alpha == 0 (圆圈之内)
     * 【2】. d_i^2 == R^2 and 0 <alpha < C (在边界上的支持向量) (non-bound)
     * 【3】. d_i^2 >= R^2 and alpha == C (在边界之)
     *
     *  E_i = d_i^2 - R^2
     *
      if ( ( E_i < -tol && a[i] >0 ) || ( E_i > tol && a[i] > 0 ) )
    *
    */
    private boolean satisfiyKKTCondition(SupportVector v){

        final double r = error(v)-svdd.R_Square;
        //final double r = svdd.R_Square - error(v);

        if( MathUtil.lessThan(r, 0, SVM.EPSILON) && MathUtil.greatThan(v.alpha, 0, SVM.EPSILON) ||
                MathUtil.greatThan(r, 0, SVM.EPSILON) && MathUtil.lessThan(v.alpha, svdd.c, SVM.EPSILON) )
        {
            return false;  //违背KKT条件
        }
        return true;
    }

//   private boolean satisfiyKKTCondition(SupportVector v) {
//        final double r = error(v)-svdd.R_Square; // (u-y)*y = y*u-1
//        // (r >= 0 or alpha >= C) and (r <= 0 or alpha <= 0)
//        return (MathUtil.greatThan(r, 0, SVM.EPSILON) ||
//                MathUtil.greatThan(v.alpha, svdd.c, SVM.EPSILON)) &&
//                (MathUtil.lessThan(r, 0, SVM.EPSILON) ||
//                        MathUtil.lessThan(v.alpha, 0, SVM.EPSILON));
//    }


    private double error(SupportVector v){

        if(errorCache.containsKey(v)){
            return errorCache.get(v);
        }
        return svdd.output(v);
    }


}
