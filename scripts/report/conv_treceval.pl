#!/usr/bin/perl -w
#
# This code is released under the
# Apache License Version 2.0 http://www.apache.org/licenses/.
#
# (c) Leonid Boytsov, http://boytsov.info
#
# For details see:
#   Leonid Boytsov, Anna Belova, Peter Westfall, 2013, 
#   Deciding on an Adjustment for Multiplicity in IR Experiments.
#   In Proceedings of SIGIR 2013.
#

sub Usage {
    my $msg = shift;
    print STDERR "$msg\n";
    print STDERR "Usage $0 <metric type: err, ndcg> <input file register: the list of input files> <output file> <optional: compute median?>\n"; 
    die();
};

my $metric       = $ARGV[0] or Usage("Metric type is missing");
my $FileRegister = $ARGV[1] or Usage("File register is missing");
my $OutputFile   = $ARGV[2] or Usage("Output file is missing");
my $ComputeMedian= $ARGV[3];

$ComputeMedian = 0 if (!defined($ComputeMedian));

my $col = 0;

open IF, "<$FileRegister" or die("Cannot open '$FileRegister' for reading");

my $ncol = 0;

open OF, ">$OutputFile" or die("Cannot open '$OutputFile' for writing");

my @DataMatrix;

while (<IF>) {
    chomp;
    my $InpFile = $_;
    next if ($InpFile =~ /^\s*$/); 

    my $DataRef;

    ($ncol, $DataRef) = ProcFile($InpFile, $metric, $ncol);

    push(@DataMatrix, $DataRef);
}

OutputData(\@DataMatrix, $ComputeMedian, $ncol, \*OF);

close IF or die("Cannot close $FileRegister");
close OF or die("Cannot close $OutputFile");

sub ProcFile {
    my ($InpFile, $metric, $PrevNcol) = @_;

    my $nc = 0;

    open F, "<$InpFile" or die("Cannot open $InpFile");

    my @Data;

    while(<F>) {
        my $line = $_;
        next if ($line =~ /^\s*$/); 
        chomp $line;
        my @DataArr = split(/\s+/, $line);
        next if ($DataArr[0] ne $metric);
        next if ($DataArr[1] eq "all");
        ++$nc;
        push(@Data, $DataArr[2]);
    }

    close F or die("Cannot close $InpFile");

    if ($PrevNcol && $nc != $PrevNcol) {
        die("# of rows differs from previous files in the list, previously seen $PrevNcol topic rows, in $InpFile there are $nc topic rows");
    }

    #print "$InpFile -> $DataCol $nc $PrevNcol\n";

    return ($nc, \@Data);
}

sub OutputData {
    my ($DataMatrixRef, $ComputeMedian, $ColQty, $OutFile) = @_;

    my @DataMatrix = @$DataMatrixRef;

    my $nr = scalar(@DataMatrix);

    if ($ComputeMedian) {
        my @Median;
        for (my $nc = 0; $nc < $ColQty; ++$nc) {
            my @Elem;

            for (my $i = 0; $i < $nr; ++$i) {
                push(@Elem, $DataMatrix[$i]->[$nc]);
            }

            push(@Median, Median(\@Elem));
        }

        unshift(@DataMatrix, \@Median);
        ++$nr;
    }

    for (my $i = 0; $i < $nr; ++$i) {
        for (my $nc = 0; $nc < $ColQty; ++$nc) {
            if ($nc) {
                print $OutFile "\t" or die("Cannot output data");
            }
            print $OutFile $DataMatrix[$i]->[$nc] or die("Cannot output data");
        }

        print $OutFile "\n" or die("Cannot output data");
    }
}

sub Median {
    my $ArrRef = shift;

    my @a = sort { $a <=> $b } @$ArrRef;
    my $N = scalar(@a);
    my $m = int($N/2);
    if ($N % 2 != 0) {
        return $a[$m];
    }
    return undef if (!$N);

    return ($a[$m] + $a[$m - 1]) / 2;
}
