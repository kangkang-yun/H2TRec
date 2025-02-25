package net.librec;

import net.librec.conf.Configuration;
import net.librec.conf.Configured;
import net.librec.data.model.TextDataModel;
import net.librec.eval.EvalContext;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.ranking.NormalizedDCGEvaluator;
import net.librec.eval.rating.MAEEvaluator;
import net.librec.job.RecommenderJob;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.ItemKNNRecommender;
import net.librec.recommender.cf.ranking.WRMFRecommender;
import net.librec.similarity.CosineSimilarity;
import net.librec.similarity.RecommenderSimilarity;

import java.io.PrintWriter;

public class MainTest {

    public static void main(String[] args) throws Exception {

//        // build data model
//        Configuration conf = new Configuration();
//        conf.set("dfs.data.dir", args[0]);
//        conf.set("dfs.data.dir", args[1]);
//
//        TextDataModel dataModel = new TextDataModel(conf);
//        dataModel.buildDataModel();
//
//        // build recommender context
//        RecommenderContext context = new RecommenderContext(conf, dataModel);
//
//        // build similarity
//        conf.set("rec.recommender.similarity.key" ,"item");
//        conf.setBoolean("rec.recommender.isranking", true);
//        conf.setInt("rec.similarity.shrinkage", 10);
//        RecommenderSimilarity similarity = new CosineSimilarity();
//        similarity.buildSimilarityMatrix(dataModel);
//        context.setSimilarity(similarity);
//
//        // build recommender
//        conf.set("rec.neighbors.knn.number", "200");
//        Recommender recommender = new WRMFRecommender();
//        recommender.setContext(context);
//
//        // run recommender algorithm
//        recommender.train(context);
//
//        // evaluate the recommended result
//        EvalContext evalContext = new EvalContext(conf, recommender, dataModel.getTestDataSet(), context.getSimilarity().getSimilarityMatrix(), context.getSimilarities());
//        RecommenderEvaluator ndcgEvaluator = new NormalizedDCGEvaluator();
//        ndcgEvaluator.setTopN(10);
//        double ndcgValue = ndcgEvaluator.evaluate(evalContext);
//        System.out.println("ndcg:" + ndcgValue);


        // revise the properties of target wide model (trustsvd, socialmf, trustmf, etc.)
        // path: librec/core/target/classes/rec/cotext/rating/*-test.properties
        Configuration.Resource resource = new Configuration.Resource("rec/context/rating/trustsvd-test.properties");
        Configuration conf = new Configuration();
        conf.addResource(resource);

        //conf.set("dfs.data.dir", "D:\\PyCharmProjects\\librec-3.0.0\\data"); // input data dir, default: data
        //conf.set("dfs.result.dir", "D:\\PyCharmProjects\\librec-3.0.0\\result_ciao"); // output result dir: revise based on different dataset_name
        //conf.set("data.model.splitter", "testset");
        //conf.set("data.input.path", "Ciao\\new_train_set_filter5.txt");
        //conf.set("data.testset.path", "Ciao\\new_test_set_filter5.txt");
        //conf.set("data.appender.path", "Ciao\\trust_data.txt");
//        conf.set("data.input.path", args[0]);
//        conf.set("data.testset.path", args[1]);
//        conf.set("rec.recommender.isranking", "true");
//        conf.set("rec.recommender.ranking.topn", "10");


        RecommenderJob job = new RecommenderJob(conf);

//        if (args[0].equals("--train")){
//        job.runTrainJob();}
//        if (args[0].equals("--test")){
//            job.runTestJob();}

        job.runJob();
    }
}
