time python nonverbal_communication_analysis/m3_DataPreprocessing/data_cleaning.py $1 -op
echo "Cleaning step: [DONE]"
time python nonverbal_communication_analysis/m4_DataProcessing/data_processing.py $1 -op -vid
echo "Processing step: [DONE]"
time python nonverbal_communication_analysis/m5_DataAggregator/feature_data_aggregator.py $1 -op -vid
echo "Merging step: [DONE]"