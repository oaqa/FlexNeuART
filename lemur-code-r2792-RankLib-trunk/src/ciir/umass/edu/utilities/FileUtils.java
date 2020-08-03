/*===============================================================================
 * Copyright (c) 2010-2015 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.utilities;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;


/**
 * This class provides some file processing utilities such as read/write files, obtain files in a
 * directory...
 * @author Van Dang
 * @version 1.3 (July 29, 2008)
 */
public class FileUtils {
	public static BufferedReader smartReader(String inputFile) throws IOException {
		return smartReader(inputFile, "UTF-8");
	}
	public static BufferedReader smartReader(String inputFile, String encoding) throws IOException {
		InputStream input = new FileInputStream(inputFile);
		if (inputFile.endsWith(".gz")) {
			input = new GZIPInputStream(input);
		}
		return new BufferedReader(new InputStreamReader(input, encoding));
	}

	/**
	 * Read the content of a file.
	 * @param filename The file to read.
	 * @param encoding The encoding of the file.
	 * @return The content of the input file.
	 */
	public static String read(String filename, String encoding) 
	{
		//String content = "";
		StringBuffer content = new StringBuffer();
		try (BufferedReader in = smartReader(filename, encoding)) {
			char[] newContent = new char[40960];
			int numRead=-1;
			while((numRead=in.read(newContent)) != -1)
			{
			    //content += new String(newContent, 0, numRead);
				content.append (new String(newContent, 0, numRead));
			}
		}
		catch(Exception e)
		{
		    //content = "";
                    content = new StringBuffer();
		}
		//return content;
                return content.toString();
	}
	
	public static List<String> readLine(String filename, String encoding) 
	{
		List<String> lines = new ArrayList<String>();
		try {
			String content = "";
			BufferedReader in = smartReader(filename, encoding);
			
			while((content = in.readLine()) != null)
			{
				content = content.trim();
				if(content.length() == 0)
					continue;
				lines.add(content);
			}
			in.close();
		}
		catch(Exception ex)
		{
			throw RankLibError.create(ex);
		}
		return lines;
	}
	/**
	 * Write a text to a file.
	 * @param filename The output filename.
	 * @param encoding The encoding of the file.
	 * @param strToWrite The string to write.
	 * @return TRUE if the procedure succeeds; FALSE otherwise.
	 */
	public static boolean write(String filename, String encoding, String strToWrite) 
	{
		BufferedWriter out = null;
		try{
			
			out = new BufferedWriter(
			          new OutputStreamWriter(new FileOutputStream(filename), encoding));
			out.write(strToWrite);
			out.close();
		}
		catch(Exception e)
		{
			return false;
		}
		return true;
	}
	/**
	 * Get all file (non-recursively) from a directory.
	 * @param directory The directory to read.
	 * @return A list of filenames (without path) in the input directory.
	 */
	public static String[] getAllFiles(String directory)
	{
		File dir = new File(directory);
		String[] fns = dir.list();
		return fns;
	}
	/**
	 * Get all file (non-recursively) from a directory.
	 * @param directory The directory to read.
	 * @return A list of filenames (without path) in the input directory.
	 */
	public static List<String> getAllFiles2(String directory)
	{
		File dir = new File(directory);
		String[] fns = dir.list();
		List<String> files = new ArrayList<String>();
		if(fns != null)
			for(int i=0;i<fns.length;i++)
				files.add(fns[i]);
		return files;
	}

	public static String getFileName(String pathName)
	{
		int idx1 = pathName.lastIndexOf("/");
		int idx2 = pathName.lastIndexOf("\\");
		int idx = (idx1 > idx2)?idx1:idx2;
		return pathName.substring(idx+1);
	}
	public static String makePathStandard(String directory)
	{
		String dir = directory;
		char c = dir.charAt(dir.length()-1);
		if(c != '/' && c != '\\')
                    //- I THINK we want File.separator (/ or \) instead of 
                    //  File.pathSeparator (: or ;) here.  Maybe needed for Analyzer?
		    //dir += File.pathSeparator;
		    dir += File.separator;
		return dir;
	}
}
