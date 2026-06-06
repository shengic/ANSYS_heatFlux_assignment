Public Sub TestProgressBar()
'Dim oUserform As UserForm1
Dim lrow As Long
Dim icount As Integer
Dim itotalcount As Integer
    itotalcount = 0
    For lrow = 1 To 5
        For icount = 1 To 1000
            itotalcount = itotalcount + 1
            Cells(lrow, 1).Value = icount
            Call UpdateProgress(itotalcount / 5000)
        Next icount
    Next lrow
End Sub

Public Sub UpdateProgress(ByVal sngPercentage As Single)
    UserForm1.lblDescription.Caption = VBA.Int(sngPercentage * 100) & "% Completed"
    UserForm1.lblBar.Width = (sngPercentage * 195)
    DoEvents
End Sub