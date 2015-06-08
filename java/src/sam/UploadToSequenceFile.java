package sam;



import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import com.google.common.io.Files;


public class UploadToSequenceFile {
	
	public static ArrayList<File> getFiles(String path) {
		ArrayList<File> files = new ArrayList<File>();
		
		File file = new File(path);
		if (file.isDirectory()) {
			files.addAll(Arrays.asList(file.listFiles()));
		} else if (file.isFile()) {
			files.add(file);
		}

		return files;
	}
	
	public static void convertToSequenceFile(String path, String sequenceFilePath) throws IOException {
		ArrayList<File> files = UploadToSequenceFile.getFiles(path);
        Configuration conf = new Configuration();
        conf.set("fs.defaultFS", "hdfs://mesos-master");
        
        // workaround for issue: http://stackoverflow.com/questions/17265002/hadoop-no-filesystem-for-scheme-file
        conf.set("fs.hdfs.impl", 
                org.apache.hadoop.hdfs.DistributedFileSystem.class.getName()
            );
        conf.set("fs.file.impl",
            org.apache.hadoop.fs.LocalFileSystem.class.getName()
        );
        
        
        SequenceFile.Writer writer = null;
        try{

            Path seqFilePath = new Path(sequenceFilePath);
            writer = SequenceFile.createWriter(conf, SequenceFile.Writer.file(seqFilePath), 
            			SequenceFile.Writer.keyClass(Text.class),
            			SequenceFile.Writer.valueClass(BytesWritable.class));
            for (File file : files) {
                byte buffer[] = Files.toByteArray(file);
            	writer.append(new Text(file.getName()), new BytesWritable(buffer));
            }
            writer.close();
            
        }catch (Exception e) {
            System.out.println("Exception MESSAGES = "+e.getMessage());
        } finally {
            IOUtils.closeStream(writer);
        }
		
	}


    public static void main(String args[]) throws IOException {
        if (args.length != 2) {
            System.err.println(
              "Usage: UploadToSequenceFile <local_folder_or_file> <hdfs_sequence_file_path>");
            System.exit(1);
          }
    	
    	String path = args[0];
    	String sequenceFilePath = args[1];
    	UploadToSequenceFile.convertToSequenceFile(path, sequenceFilePath);
    	
    	
    }
}
