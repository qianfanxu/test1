package SMOTest2.svm;


import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

import SMOTest2.util.MathUtil;

/**
 * Created by xuqf on 2017/2/15.
 */
public class SMO {

    // random select the Vector to train
    private static final Random random = new Random();

    // svm to be trained
    private SVM svm;

    //error cache
    private Map<SupportVector, Double> errorCache = new HashMap<SupportVector, Double>();  //[???] 为什么HashMap 要使用包装类

    private SMO(SVM svm){
        this.svm = svm;
    }

    public static void train(SVM svm){

        SMO smo = new SMO(svm);
        smo.train();

    }


    private void train(){

        int numChanged = 0;
        boolean examineAll = true;

        // All data set and non-bound data set is alternatively progress
        while(examineAll || numChanged > 0){  // search for all and no v is changed.

            numChanged = 0;
            for(SupportVector v: svm.vectors){

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
        final int n = svm.vectors.size();
        final int pos = random.nextInt(n);
        for( SupportVector v1: svm.vectors.subList(pos, n) ){
            if(!v1.bound && takeStep(v1,v)){
                return true;
            }
        }

        for( SupportVector v1: svm.vectors.subList(0, pos) ){
            if(!v1.bound && takeStep(v1,v)){
                return true;
            }
        }

        for( SupportVector v1: svm.vectors.subList(0, pos) ){
            if(v1.bound && takeStep(v1,v)){
                return true;
            }
        }


        for( SupportVector v1: svm.vectors.subList(0, pos) ){
            if(v1.bound && takeStep(v1,v)){
                return true;
            }
        }

        return false;
    }


    //找到步长最大的E2：即 |E1-E2|
    private SupportVector secondChoice(double error){

        Entry<SupportVector,Double> best = null;

        if(error > 0){
            for(Entry<SupportVector,Double> entry: errorCache.entrySet()){
                if(best == null || entry.getValue() < best.getValue() )
                    best = entry;
            }
        }
        else{

            for(Entry<SupportVector,Double> entry: errorCache.entrySet()){
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
        final double y1 = v1.y, y2 = v2.y;

        double l, h;

        // 确定 L H 上下界 [???] 为什么取这个最大最小值
        if(y1 != y2) {
            l = Math.max(0, alpha2_old - alpha1_old);
            h = Math.min(svm.c, svm.c + alpha2_old - alpha1_old);
        } else {

            l = Math.max(0, alpha2_old + alpha1_old - svm.c);
            h = Math.min(svm.c, alpha2_old + alpha1_old);
        }
        if(l == h)
            return false;

        // 对目标函数求导得出，对a2的一个更新
        final double k11 = svm.kernel.getValue(v1.x,v1.x);
        final double k12 = svm.kernel.getValue(v1.x,v2.x);
        final double k22 = svm.kernel.getValue(v2.x,v2.x);

        // 对a2进行剪辑
        final double s = y1*y2;
        final double e1 = error(v1), e2 = error(v2);

        final double eta = k11 + k22 - 2*k12;

        if( eta < 0 ){
            throw new RuntimeException("violate Mercer's condition");
        }

        if( eta == 0 ){
            System.out.println("debug: eta == 0 ");
            return false;
        }

        v2.alpha = alpha2_old + y2 * (e1 - e2) / eta;
        //进行剪辑
        v2.alpha = MathUtil.window(v2.alpha, l, h);


        if(MathUtil.equals(v2.alpha, alpha2_old, SVM.EPSILON*(v2.alpha+alpha2_old+SVM.EPSILON))) {
            // change in alpha2 was too small
            v2.alpha = alpha2_old;
            return false;
        }


        v1.alpha = alpha1_old + s*(alpha2_old-v2.alpha);
        v1.bound = MathUtil.lessThan(v1.alpha, 0, SVM.EPSILON) || MathUtil.greatThan(v1.alpha, svm.c, SVM.EPSILON);
        v2.bound = MathUtil.lessThan(v2.alpha, 0, SVM.EPSILON) || MathUtil.greatThan(v2.alpha, svm.c, SVM.EPSILON);

        // 更新阈值b
        final double b = svm.b;
        final double delta1 = y1*(v1.alpha-alpha1_old);
        final double delta2 = y2*(v2.alpha-alpha2_old);

        final double b1 = e1 + delta1*k11 + delta2*k12;
        final double b2 = e2 + delta1*k12 + delta2*k22;

        if(!v1.bound){
            svm.b += b1;
        }
        else if(!v2.bound){
            svm.b += b2;
        }
        else{
            svm.b += (b1 + b2)/2;
        }

        // 更新errorCache
        for(SupportVector v: svm.vectors){

            if( v.bound ) continue;
            if( v == v1 || v == v2 ) continue;
            final double k1 = svm.kernel.getValue(v1.x, v.x);
            final double k2 = svm.kernel.getValue(v2.x, v.x);
            double error = errorCache.get(v);
            error += delta1*k1 + delta2*k2 + b - svm.b;
            errorCache.put(v,error);
        }

        /*
        *  1. 只是储存无界（non-bound  0<ai<C）的向量
           2. 将所有做优化的 ai （joint optimization的 ai），error值全部置为0，否则移除
        *
        */

        if(!v1.bound) {
            errorCache.put(v1,0.0);
        }
        else{
            errorCache.remove(v1);
        }

        if(!v2.bound) {
            errorCache.put(v2,0.0);
        }
        else{
            errorCache.remove(v2);
        }

        return true;
    }


    /*
    *
    * 把违背KKT条件的ai作为第一个
     * 满足KKT条件的情况是：
     * 【1】. yi*f(i) -1 >= 0 and alpha == 0 (正确分类)
     * 【2】. yi*f(i) -1 == 0 and 0 <alpha < C (在边界上的支持向量) (non-bound)
     * 【3】. yi*f(i) -1 <= 0 and alpha == C (在边界之间)
     *
     if ( ( y[i] * Ei < -tol && a[i] < C ) || ( y[i] * Ei > tol && a[i] > 0 ) )
    *
    * (u-y)*y = y*u-1
    */
    private boolean satisfiyKKTCondition(SupportVector v){

        final double r = error(v)*v.y;

        if( MathUtil.lessThan(r, 0, SVM.EPSILON) && MathUtil.lessThan(v.alpha,svm.c,SVM.EPSILON) ||
            MathUtil.greatThan(r, 0, SVM.EPSILON) && MathUtil.greatThan(v.alpha,0, SVM.EPSILON) )
        {
            return false;  //违背KKT条件
        }
        return true;
    }

//   private boolean satisfiyKKTCondition(SupportVector v) {
//        final double r = error(v) * v.y; // (u-y)*y = y*u-1
//        // (r >= 0 or alpha >= C) and (r <= 0 or alpha <= 0)
//        return (MathUtil.greatThan(r, 0, SVM.EPSILON) ||
//                MathUtil.greatThan(v.alpha, svm.c, SVM.EPSILON)) &&
//                (MathUtil.lessThan(r, 0, SVM.EPSILON) ||
//                        MathUtil.lessThan(v.alpha, 0, SVM.EPSILON));
//    }


    private double error(SupportVector v){

        if(errorCache.containsKey(v)){
            return errorCache.get(v);
        }
        return svm.output(v) - v.y;
    }


    /*
    private double output(SupportVector v1){

        // $u = sigmaOf(i)*ai*yi*K(xi, x) - b$
        double sum = svm.b;
        for(SupportVector v: svm.vectors){

            if( v.alpha <= SVM.EPSILON){
                continue;
            }
            sum += (v.alpha)*(v.y)*(svm.kernel.getValue(v.x, v1.x));
        }
        return sum;
    }
    */

}
