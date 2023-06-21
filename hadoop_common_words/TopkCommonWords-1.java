
import java.io.IOException;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Set;
import java.util.HashSet;
import java.util.Map;
import java.util.HashMap;
import java.util.PriorityQueue;
import java.util.Stack;
import java.util.AbstractMap.SimpleEntry;
import java.util.Comparator;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class TopkCommonWords {

  public static class TopKMapper 
       extends Mapper<Object, Text, Text, IntWritable>{ // specify input key, input value, output key, and output value
    
    private final Text fileSource = new Text(); // Text variable to keep track of the source file of each split
    private Set<String> stopwords = new HashSet<>(); // create a new HashSet to read stopwords into
    private Map<String, Integer> countMap = new HashMap<String, Integer>(); // create a new dictionary to aggregate wordcounts

    @Override
    public void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration(); // get configuration
      String inputPath1 = conf.get("inputPath1"); // read inputPath1 from conf
      Path path = new Path(inputPath1);
      String inputFile1 = path.getName(); 
      String inputFile = ((FileSplit) context.getInputSplit()).getPath().getName(); // get the input file for this split (each split receives from the same path)
      if (inputFile.equals(inputFile1)) { // inputs from input1 are labelled A, and those from input2 are labelled B
        fileSource.set(" A");
      }
      else {fileSource.set(" B");}

      String stopwordsFilePath = conf.get("stopwords.txt"); // access stopwords data from conf
      if (stopwordsFilePath != null) {
        File stopwordsFile = new File(stopwordsFilePath);
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(stopwordsFile))); // reading in stopwords and storing in stopwords set
        String line;
        while ((line = reader.readLine()) != null) { // for each line in stopwords.txt...
          stopwords.add(line.trim());  // trim to match StringTokenizer format when comparing in map stage, then add to stopwords set
        }
        reader.close(); // close reader when done
      }
    }

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException { 
      StringTokenizer itr = new StringTokenizer(value.toString()); // creating a StringTokenizer object to tokenize the input text - NOTE: The tokenizer uses the default delimiter set, which is " \t\n\r\f"
      while (itr.hasMoreTokens()) { // iterating over each token
        String wordStr = itr.nextToken();
        if ( (wordStr.length() > 4) && (!stopwords.contains(wordStr)) ) { // dont bother with stopwords or those less than 5 characters
          countMap.put(wordStr, countMap.getOrDefault(wordStr, 0) + 1); // update wordcount in the corresponding dictionary (in-mapper aggregation)
        }
      }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      for (Map.Entry<String,Integer> entry : countMap.entrySet()) {
        context.write(new Text(entry.getKey() + fileSource.toString()), new IntWritable(entry.getValue())); // write to context - making sure to cast to correct types... (emit)
      }
      countMap.clear(); // clear map at the end of map step
    }
  }

  public static class TopKReducer
       extends Reducer<Text, IntWritable, IntWritable, Text> { // specifying the input key, input value, output key, and output value
    private PriorityQueue<SimpleEntry<Integer, String>> topWords = new PriorityQueue<SimpleEntry<Integer, String>>(new Comparator<SimpleEntry<Integer, String>>() { // make PQ to sort results according to desired output
      public int compare(SimpleEntry<Integer, String> a, SimpleEntry<Integer, String> b) { // declare our own custom comparator...
        int keyComparison = a.getKey().compareTo(b.getKey());
        if (keyComparison != 0) { 
          return a.getKey().compareTo(b.getKey());  // first try sort by wordcount...
        }
        return  b.getValue().compareTo(a.getValue()); // break wordcount ties on String values...
      }});
    private Map<String, Integer> allcountsMap = new HashMap<String, Integer>(); // new dictionary that holds counts from both inputs
    private int k; // k is private, dont touch that 😠

    @Override
    public void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      k = conf.getInt("k",10); // read in k, defaults to 10 if not provided
    }

    public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException { 
      int sum = 0; // initializing the sum variable to 0
      for (IntWritable val : values) { // iterating over each value for the current key
        sum += val.get(); // adding the value to the sum variable    
      }
      
      String keyStr = key.toString(); // NOTE: keyStr will look like: "EXAMPLE A" or "EXAMPLE B" depending on source file
      allcountsMap.put(keyStr, sum); // put the summed count into our combined dictionary

      if (keyStr.charAt(keyStr.length() - 1) == 'A'){ // if its from A...
        String keyStrB = keyStr.replaceAll(" A"," B"); 
        if (allcountsMap.containsKey(keyStrB)){ // have we seen this word from B yet? if yes then...
          topWords.add(new SimpleEntry<Integer, String> // take the minimum occurances between the two and shave off the filesource label, then add to our topWords
            (Math.min(allcountsMap.get(keyStrB),allcountsMap.get(keyStr)), keyStrB.substring(0,keyStrB.length()-2))); 
          if (topWords.size() > k) {
            topWords.poll(); // drop the smallest entry if we go past k
          }
        }
      } else { // key comes from input B... (same as above)
        String keyStrA = keyStr.replaceAll(" B"," A");
        if (allcountsMap.containsKey(keyStrA)){
          topWords.add(new SimpleEntry<Integer, String>(Math.min(allcountsMap.get(keyStrA),allcountsMap.get(keyStr)), keyStrA.substring(0,keyStrA.length()-2)));
          if (topWords.size() > k) {
            topWords.poll();
          }
        }
      }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      Stack<SimpleEntry<Integer, String>> result = new Stack<>(); // reversing the order of output by pouring into a stack (could've used DEQ instead...)
      while (!topWords.isEmpty()) {
        result.push(topWords.poll());
      }
      while (!result.empty()) {
        SimpleEntry<Integer, String> pair = result.pop();
        context.write(new IntWritable(pair.getKey()), new Text(pair.getValue())); // write everything out of the stack
      }
      allcountsMap.clear(); // clear dictionary after reduce stage is complete
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration(); // create our configuration object
    conf.set("inputPath1", args[0]); // set inputs to config so that we can retrieve the paths in our mapper later
    conf.set("inputPath2", args[1]); 
    conf.set("stopwords.txt", args[2]); // set stopwords file path
    conf.setInt("k", Integer.parseInt(args[4])); // set k value

    Job job = Job.getInstance(conf, "top k words");
    job.setJarByClass(TopkCommonWords.class); // feed in all the relevant classes
    job.setMapperClass(TopKMapper.class);
    job.setReducerClass(TopKReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);

    FileInputFormat.setInputPaths(job, new Path(args[0]), new Path(args[1])); // set up inputs
    FileOutputFormat.setOutputPath(job, new Path(args[3]));

    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
