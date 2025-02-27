/**
 * Copyright (C) 2016 LibRec
 * <p>
 * This file is part of LibRec.
 * LibRec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * LibRec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with LibRec. If not, see <http://www.gnu.org/licenses/>.
 */
package net.librec.job;

import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.DataModel;
import net.librec.data.DataSplitter;
import net.librec.data.splitter.KCVDataSplitter;
import net.librec.data.splitter.LOOCVDataSplitter;
import net.librec.eval.EvalContext;
import net.librec.eval.Measure;
import net.librec.eval.Measure.MeasureValue;
import net.librec.eval.RecommenderEvaluator;
import net.librec.filter.RecommendedFilter;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.DataSet;
import net.librec.math.structure.SymmMatrix;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.item.RecommendedItem;
import net.librec.similarity.RecommenderSimilarity;
import net.librec.util.DriverClassUtil;
import net.librec.util.FileUtil;
import net.librec.util.JobUtil;
import net.librec.util.ReflectionUtil;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * RecommenderJob
 *
 * @author WangYuFeng
 */
public class RecommenderJob {
    /**
     * LOG
     */
    protected final Log LOG = LogFactory.getLog(RecommenderJob.class);

    private Configuration conf;

    private DataModel dataModel;

    private Map<String, List<Double>> cvEvalResults;

    private Map<MeasureValue, Double> evaluatedMap;

    private Recommender recommender;

    public RecommenderJob(Configuration conf) {
        this.conf = conf;
        Long seed = conf.getLong("rec.random.seed");
        if (seed != null) {
            Randoms.seed(seed);
        }
        setJobId(JobUtil.generateNewJobId());
    }

    /**
     * run Job
     *
     * @throws LibrecException        If an LibrecException error occurs.
     * @throws ClassNotFoundException if can't find the class of filter
     * @throws IOException            If an I/O error occurs.
     */
    public void runJob() throws LibrecException, ClassNotFoundException, IOException {
        executeRecommenderJob();
    }

//    /**
//     * execute Recommender Job
//     *
//     * @throws LibrecException        If an LibrecException error occurs.
//     * @throws ClassNotFoundException if can't find the class of filter
//     * @throws IOException            If an I/O error occurs.
//     */
//    @SuppressWarnings("unchecked")
//    private void executeRecommenderJob() throws ClassNotFoundException, LibrecException, IOException {
//        generateDataModel();
//        RecommenderContext context = new RecommenderContext(conf, dataModel);
//        generateSimilarity(context);
//        Recommender recommender = ReflectionUtil.newInstance((Class<Recommender>) getRecommenderClass(), conf);
//        recommender.train(context);
//
//        executeEvaluator(recommender, context);
////        List<RecommendedItem> recommendedItemsList = recommender.getRecommendedList(recommendedList);
////        recommendedItemsList = filterResult(recommendedItemsList);
////        saveResult(recommendedItemsList);
//    }

    /**
     * execute Recommender Job
     *
     * @throws LibrecException        If an LibrecException error occurs.
     * @throws ClassNotFoundException if can't find the class of filter
     * @throws IOException            If an I/O error occurs.
     */
    @SuppressWarnings("unchecked")
    private void executeRecommenderJob() throws ClassNotFoundException, LibrecException, IOException {
        generateDataModel();
        recommender = ReflectionUtil.newInstance((Class<Recommender>) getRecommenderClass(), conf);
        RecommenderContext context = new RecommenderContext(conf);
        cvEvalResults = new HashMap<>();
        while(dataModel.hasNextFold()) {
            dataModel.nextFold();
            context.setDataModel(dataModel);
            generateSimilarity(context);
            recommender.train(context);
            executeEvaluator(recommender, context);
        }
        printCVAverageResult();
        boolean isRanking = conf.getBoolean("rec.recommender.isranking");
        List<RecommendedItem> recommendedList = null;
        if (isRanking){
            recommendedList = recommender.getRecommendedList(recommender.recommendRank());
        } else {
            recommendedList = recommender.getRecommendedList(recommender.recommendRating(context.getDataModel().getTestDataSet()));
        }
        recommendedList = filterResult(recommendedList);
        saveResult(recommendedList);
    }

    /**
     * Generate data model.
     *
     * @throws ClassNotFoundException
     * @throws IOException
     * @throws LibrecException
     */
    @SuppressWarnings("unchecked")
    private void generateDataModel() throws ClassNotFoundException, IOException, LibrecException {
        if (null == dataModel) {
            dataModel = ReflectionUtil.newInstance((Class<DataModel>) this.getDataModelClass(), conf);
        }
        dataModel.buildDataModel();
    }

    /**
     * Generate similarity.
     *
     * @param context recommender context
     */
    private void generateSimilarity(RecommenderContext context) {
        String[] similarityKeys = conf.getStrings("rec.recommender.similarities");
        if (similarityKeys != null && similarityKeys.length > 0) {
            for (int i = 0; i < similarityKeys.length; i++) {
                if (getSimilarityClass() != null) {
                    RecommenderSimilarity similarity = ReflectionUtil.newInstance(getSimilarityClass(), conf);
                    conf.set("rec.recommender.similarity.key", similarityKeys[i]);
                    similarity.buildSimilarityMatrix(dataModel);
                    if (i == 0) {
                        context.setSimilarity(similarity);
                    }
                    context.addSimilarities(similarityKeys[i], similarity);
                }
            }
        }
    }

    /**
     * Filter the results.
     *
     * @param recommendedList list of recommended items
     * @return recommended List
     * @throws ClassNotFoundException
     * @throws IOException
     */
    private List<RecommendedItem> filterResult(List<RecommendedItem> recommendedList) throws ClassNotFoundException, IOException {
        if (getFilterClass() != null) {
            RecommendedFilter filter = ReflectionUtil.newInstance(getFilterClass(), null);
            recommendedList = filter.filter(recommendedList);
        }
        return recommendedList;
    }

    /**
     * Execute evaluator.
     *
     * @throws LibrecException        if error occurs
     * @throws IOException            if I/O error occurs
     * @throws ClassNotFoundException if class not found error occurs
     */
    private void executeEvaluator(Recommender recommender, RecommenderContext context) throws ClassNotFoundException, IOException, LibrecException {
        if (conf.getBoolean("rec.eval.enable")) {
            DataSet dataSet =  dataModel.getTestDataSet();
            String[] similarityKeys = conf.getStrings("rec.recommender.similarities");
            EvalContext evalContext = null;
            if (similarityKeys != null && similarityKeys.length > 0) {
                SymmMatrix similarityMatrix = context.getSimilarity().getSimilarityMatrix();
                Map<String, RecommenderSimilarity> similarities = context.getSimilarities();
                evalContext = new EvalContext(conf, recommender, dataSet, similarityMatrix, similarities);
            } else {
                evalContext = new EvalContext(conf, recommender, dataSet);
            }


            String[] evalClassKeys = conf.getStrings("rec.eval.classes");
            if (evalClassKeys != null && evalClassKeys.length > 0) {// Run the evaluator which is
                // designated.
                for (int classIdx = 0; classIdx < evalClassKeys.length; ++classIdx) {
                    RecommenderEvaluator evaluator = ReflectionUtil.newInstance(getEvaluatorClass(evalClassKeys[classIdx]), null);
                    evaluator.setTopN(conf.getInt("rec.recommender.ranking.topn", 10));

                    double evalValue = evaluator.evaluate(evalContext);
                    LOG.info("Evaluator info:" + evaluator.getClass().getSimpleName() + " is " + evalValue);
                    collectCVResults(evaluator.getClass().getSimpleName(), evalValue);
                }
            } else {// Run all evaluators
                evaluatedMap = new HashMap<>();
                boolean isRanking = conf.getBoolean("rec.recommender.isranking");
                int topN = 10;
                if (isRanking) {
                    topN = conf.getInt("rec.recommender.ranking.topn", 10);
                    if (topN <= 0) {
                        throw new IndexOutOfBoundsException("rec.recommender.ranking.topn should be more than 0!");
                    }
                }
                List<MeasureValue> measureValueList = Measure.getMeasureEnumList(isRanking, topN);
                if (measureValueList != null) {
                    for (MeasureValue measureValue : measureValueList) {
                        RecommenderEvaluator evaluator = ReflectionUtil
                                .newInstance(measureValue.getMeasure().getEvaluatorClass());
                        if (isRanking && measureValue.getTopN() != null && measureValue.getTopN() > 0) {
                            evaluator.setTopN(measureValue.getTopN());
                        }
                        double evaluatedValue = evaluator.evaluate(evalContext);
                        evaluatedMap.put(measureValue, evaluatedValue);
                    }
                }
                if (evaluatedMap.size() > 0) {
                    for (Map.Entry<MeasureValue, Double> entry : evaluatedMap.entrySet()) {
                        String evalName = null;
                        if (entry != null && entry.getKey() != null) {
                            if (entry.getKey().getTopN() != null && entry.getKey().getTopN() > 0) {
                                LOG.info("Evaluator value:" + entry.getKey().getMeasure() + " top " + entry.getKey().getTopN() + " is " + entry.getValue());
                                evalName = entry.getKey().getMeasure() + " top " + entry.getKey().getTopN();
                            } else {
                                LOG.info("Evaluator value:" + entry.getKey().getMeasure() + " is " + entry.getValue());
                                evalName = entry.getKey().getMeasure() + "";
                            }
                            if (null != cvEvalResults) {
                                collectCVResults(evalName, entry.getValue());
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Save result.
     *
     * @param recommendedList list of recommended items
     * @throws LibrecException        if error occurs
     * @throws IOException            if I/O error occurs
     * @throws ClassNotFoundException if class not found error occurs
     */
    public void saveResult(List<RecommendedItem> recommendedList) throws LibrecException, IOException, ClassNotFoundException {
        if (recommendedList != null && recommendedList.size() > 0) {
            // make output path
            String algoSimpleName = DriverClassUtil.getDriverName(getRecommenderClass());
            String outputPath = conf.get("dfs.result.dir") + "/" + conf.get("data.input.path") + "-" + algoSimpleName + "-output/" + algoSimpleName;
            if (null != dataModel && (dataModel.getDataSplitter() instanceof KCVDataSplitter || dataModel.getDataSplitter() instanceof LOOCVDataSplitter) && null != conf.getInt("data.splitter.cv.index")) {
                outputPath = outputPath + "-" + String.valueOf(conf.getInt("data.splitter.cv.index"));
            }
            LOG.info("Result path is " + outputPath);
            // convert itemList to string
            StringBuilder sb = new StringBuilder();
            for (RecommendedItem recItem : recommendedList) {
                String userId = recItem.getUserId();
                String itemId = recItem.getItemId();
                String value = String.valueOf(recItem.getValue());
                sb.append(userId).append(",").append(itemId).append(",").append(value).append("\n");
            }
            String resultData = sb.toString();
            // save resultData
            try {
                FileUtil.writeString(outputPath, resultData);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Print the average evaluate results when using cross validation.
     */
    private void printCVAverageResult() {
        DataSplitter splitter = dataModel.getDataSplitter();
        if (splitter != null && (splitter instanceof KCVDataSplitter || splitter instanceof LOOCVDataSplitter)) {
            LOG.info("Average Evaluation Result of Cross Validation:");
            for (Map.Entry<String, List<Double>> entry : cvEvalResults.entrySet()) {
                String evalName = entry.getKey();
                List<Double> evalList = entry.getValue();
                double sum = 0.0;
                for (double value : evalList) {
                    sum += value;
                }
                double avgEvalResult = sum / evalList.size();
                LOG.info("Evaluator value:" + evalName + " is " + avgEvalResult);
            }
        }
    }

    /**
     * Collect the evaluate results when using cross validation.
     *
     * @param evalName  name of the evaluator
     * @param evalValue value of the evaluate result
     */
    private void collectCVResults(String evalName, Double evalValue) {
        DataSplitter splitter = dataModel.getDataSplitter();
        if (splitter != null && (splitter instanceof KCVDataSplitter || splitter instanceof LOOCVDataSplitter)) {
            if (cvEvalResults.containsKey(evalName)) {
                cvEvalResults.get(evalName).add(evalValue);
            } else {
                List<Double> newList = new ArrayList<>();
                newList.add(evalValue);
                cvEvalResults.put(evalName, newList);
            }
        }
    }

    private void setJobId(String jobId) {
        conf.set("rec.job.id", jobId);
    }

    public void setRecommenderClass(String jobClass) {
        conf.set("rec.recommender.class", jobClass);
    }

    public void setRecommenderClass(Class<Recommender> jobClass) {
        conf.set("rec.recommender.class", jobClass.getName());
    }

    /**
     * Get data model class.
     *
     * @return {@code Class<? extends DataModel>} object
     * @throws ClassNotFoundException if the class is not found
     * @throws IOException            If an I/O error occurs.
     */
    @SuppressWarnings("unchecked")
    public Class<? extends DataModel> getDataModelClass() throws ClassNotFoundException, IOException {
        return (Class<? extends DataModel>) DriverClassUtil.getClass(conf.get("data.model.format"));
    }

    /**
     * Get similarity class
     *
     * @return similarity class object
     */
    @SuppressWarnings("unchecked")
    public Class<? extends RecommenderSimilarity> getSimilarityClass() {
        try {
            return (Class<? extends RecommenderSimilarity>) DriverClassUtil.getClass(conf.get("rec.similarity.class"));
        } catch (ClassNotFoundException e) {
            return null;
        }
    }

    /**
     * Get recommender class. {@code Recommender}.
     *
     * @return recommender class object
     * @throws ClassNotFoundException if can't find the class of recommender
     * @throws IOException            If an I/O error occurs.
     */
    @SuppressWarnings("unchecked")
    public Class<? extends Recommender> getRecommenderClass() throws ClassNotFoundException, IOException {
        return (Class<? extends Recommender>) DriverClassUtil.getClass(conf.get("rec.recommender.class"));
    }

    /**
     * Get evaluator class. {@code RecommenderEvaluator}.
     *
     * @param evalClassKey class key of the evaluator
     * @return evaluator class object
     * @throws ClassNotFoundException if can't find the class of evaluator
     * @throws IOException            If an I/O error occurs.
     */
    @SuppressWarnings("unchecked")
    public Class<? extends RecommenderEvaluator> getEvaluatorClass(String evalClassKey) throws ClassNotFoundException, IOException {
        return (Class<? extends RecommenderEvaluator>) DriverClassUtil.getClass(evalClassKey);
    }

    /**
     * Get filter class. {@code RecommendedFilter}.
     *
     * @return evaluator class object
     * @throws ClassNotFoundException if can't find the class of filter
     * @throws IOException            If an I/O error occurs.
     */
    @SuppressWarnings("unchecked")
    public Class<? extends RecommendedFilter> getFilterClass() throws ClassNotFoundException, IOException {
        return (Class<? extends RecommendedFilter>) DriverClassUtil.getClass(conf.get("rec.filter.class"));
    }

    public Map<MeasureValue, Double> getEvaluatedMap() {
        return evaluatedMap;
    }
}
